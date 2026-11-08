# =========================================================
# PROJECT B — Telco Customer Churn: Streamlit app
# Loads the five saved pipelines from models/ (no retraining),
# predicts churn on an uploaded CSV and reports all six metrics.
# =========================================================

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📶", layout="wide")


# ---------------------------------------------------------
# LOAD SAVED ARTEFACTS (cached: loaded once, never retrained)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    return meta, models


@st.cache_data
def load_saved_metrics():
    return pd.read_csv(MODEL_DIR / "metrics_project_b.csv", index_col=0)


meta, models = load_artifacts()
TARGET = meta["target"]                                   # "Churn"
FEATURES = meta["feature_columns"]                         # 19 columns, training order
NUM_COLS = meta["numeric_columns"]                         # tenure, MonthlyCharges, TotalCharges
CAT_COLS = meta["categorical_columns"]                     # 16 columns
CLASS_NAMES = {int(k): v for k, v in meta["class_names"].items()}   # {0: "No churn", 1: "Churn"}
LABELS = [0, 1]

# Category values each column was trained with (read from the saved encoder, not hard-coded)
_encoder = models["Logistic Regression"].named_steps["prep"].named_transformers_["cat"]
KNOWN_CATEGORIES = {col: set(cats.tolist()) for col, cats in zip(CAT_COLS, _encoder.categories_)}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def read_csv_safely(source):
    """Read a CSV, returning (dataframe, error message)."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return None, "The file is empty."
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        return None, f"The file could not be read as a CSV ({type(e).__name__})."
    df.columns = [str(c).strip() for c in df.columns]          # tolerate stray spaces in headers
    if df.empty:
        return None, "The file has column headers but no data rows."
    return df, None


def prepare_features(df):
    """Apply the same cleaning as training. Returns (X, kept_mask, notes)."""
    notes = []
    X = df[FEATURES].copy()

    # Text columns: strip spaces so " Yes" matches "Yes"
    for col in CAT_COLS:
        if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
            X[col] = X[col].astype(str).str.strip()

    # SeniorCitizen was trained as 0/1; also accept Yes/No
    sc = X["SeniorCitizen"].astype(str).str.strip().str.lower().map(
        {"0": 0, "1": 1, "no": 0, "yes": 1, "0.0": 0, "1.0": 1})
    X["SeniorCitizen"] = sc

    # TotalCharges: blank values belong to brand-new customers (tenure 0) -> 0, as in training
    total = pd.to_numeric(X["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
    blanks = total.isna() & X["TotalCharges"].astype(str).str.strip().eq("")
    if blanks.any():
        notes.append(f"{int(blanks.sum())} blank TotalCharges value(s) were set to 0, as in training.")
    X["TotalCharges"] = total.where(~blanks, 0.0)

    for col in ["tenure", "MonthlyCharges"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    bad = X[NUM_COLS + ["SeniorCitizen"]].isna().any(axis=1)
    if bad.any():
        notes.append(f"{int(bad.sum())} row(s) with missing or non-numeric tenure, charges or "
                     "SeniorCitizen values were skipped.")

    # Categories never seen in training would be silently ignored by the encoder, so report them
    unseen = {col: sorted(set(X.loc[~bad, col].astype(str)) - {str(v) for v in KNOWN_CATEGORIES[col]})
              for col in CAT_COLS}
    unseen = {c: v for c, v in unseen.items() if v}
    if unseen:
        notes.append("Values not seen during training (treated as unknown): "
                     + "; ".join(f"{c}: {v}" for c, v in unseen.items()))
    return X[~bad], ~bad, notes


def parse_labels(series):
    """Accept Yes/No, Churn/No churn, True/False or 1/0."""
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0,
               "churn": 1, "no churn": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def compute_metrics(y_true, y_pred, churn_proba):
    """Six metrics; Precision, Recall and F1 are for the Churn class (positive class = 1)."""
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if len(set(y_true)) == 2:                     # AUC needs both classes in the true labels
        results["AUC"] = roc_auc_score(y_true, churn_proba)
    return results


# ---------------------------------------------------------
# SIDEBAR: model choice and data source
# ---------------------------------------------------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload a test CSV", type="csv")
use_sample = st.sidebar.checkbox("Use the bundled test set instead", value=uploaded is None)

st.sidebar.download_button("Download the bundled test set",
                           data=(MODEL_DIR / "test_set.csv").read_bytes(),
                           file_name="test_set.csv", mime="text/csv")
st.sidebar.caption(f"Models trained with scikit-learn {meta['sklearn_version']} · "
                   f"running {sklearn.__version__}")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("📶 Telco Customer Churn Prediction")
st.write("Predicts whether a telecom customer will **churn** (leave) from their contract, services "
         "and billing details, using five pre-trained scikit-learn models. Upload a CSV with the 19 "
         "customer columns (and optionally the `Churn` label) to get predictions and evaluation metrics.")

tab_eval, tab_compare, tab_about = st.tabs(["Predict & evaluate", "Model comparison", "About the data"])


# ---------------------------------------------------------
# TAB 1: PREDICT & EVALUATE
# ---------------------------------------------------------
with tab_eval:
    if uploaded is not None and not use_sample:
        df, read_error = read_csv_safely(uploaded)
        source = f"uploaded file **{uploaded.name}**"
    elif use_sample:
        df, read_error = read_csv_safely(MODEL_DIR / "test_set.csv")
        source = "the bundled **test_set.csv** (1,409 held-out customers)"
    else:
        st.info("Upload a CSV in the sidebar, or tick **Use the bundled test set**.")
        st.stop()
    if read_error:
        st.error(read_error)
        st.stop()

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"The file is missing {len(missing)} required column(s): {missing}")
        st.stop()

    X, kept, notes = prepare_features(df)
    for note in notes:
        st.warning(note)
    df = df[kept.to_numpy()]
    if X.empty:
        st.error("No usable rows left after removing invalid values.")
        st.stop()

    model = models[model_name]
    y_pred = model.predict(X)
    churn_col = list(model.classes_).index(1)
    churn_proba = model.predict_proba(X)[:, churn_col]

    st.subheader(f"{model_name} on {len(X):,} customers")
    st.caption(f"Data source: {source}")

    if TARGET in df.columns:
        y_true = parse_labels(df[TARGET])
        valid = y_true.notna().to_numpy()
        if not valid.all():
            st.warning(f"{int((~valid).sum())} row(s) have a `{TARGET}` value other than Yes/No or 1/0; "
                       "they are excluded from the metrics.")
        y_true_v = y_true[valid].astype(int).to_numpy()
        y_pred_v, proba_v = y_pred[valid], churn_proba[valid]

        # Six metrics
        scores = compute_metrics(y_true_v, y_pred_v, proba_v)
        cols = st.columns(6)
        for col, (name, value) in zip(cols, scores.items()):
            col.metric(name, "n/a" if pd.isna(value) else f"{value:.4f}")
        if pd.isna(scores["AUC"]):
            st.caption("AUC needs both churners and non-churners in the uploaded labels, so it is not shown.")
        st.caption("Precision, Recall and F1 are calculated for the **Churn** class, the customers the "
                   "business wants to find. AUC uses the predicted churn probability.")

        # Confusion matrix + classification report
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_true_v, y_pred_v, labels=LABELS)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay(cm, display_labels=[CLASS_NAMES[l] for l in LABELS]).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("**Classification report**")
            report = classification_report(y_true_v, y_pred_v, labels=LABELS,
                                           target_names=[CLASS_NAMES[l] for l in LABELS],
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
    else:
        st.info(f"No `{TARGET}` column found, so only predictions are shown (metrics need true labels).")

    # Predictions table + download
    st.markdown("**Predictions**")
    out = df.copy()
    out["predicted_churn"] = [CLASS_NAMES[int(c)] for c in y_pred]
    out["churn_probability"] = churn_proba.round(4)
    st.dataframe(out, width="stretch", height=300)
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"churn_predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")


# ---------------------------------------------------------
# TAB 2: MODEL COMPARISON (saved results on the held-out test set)
# ---------------------------------------------------------
with tab_compare:
    st.subheader("All five models on the held-out test set")
    saved = load_saved_metrics()
    st.dataframe(saved.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    st.caption("Green = best value in each column. Computed once after training on the "
               "1,409-customer stratified test set; shown here without retraining.")


# ---------------------------------------------------------
# TAB 3: ABOUT THE DATA
# ---------------------------------------------------------
with tab_about:
    st.markdown(f"""
**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
7,043 customers with contract, service and billing details.

**Target:** `{TARGET}` — Yes (churned) = 1, No = 0 (about 26.5% churn).

**Preprocessing:** `customerID` dropped (an identifier, not a predictor); the 11 blank `TotalCharges`
values (customers with tenure 0, never billed) set to 0; stratified 80/20 split with random state
{meta['random_state']}. One-hot encoding of the 16 categorical columns and scaling of the 3 numeric
columns live inside each saved pipeline and were fitted on the training data only.

**Models:** Logistic Regression, Decision Tree, KNN (k = 5), Gaussian Naive Bayes and an
RBF-kernel SVM with calibrated probabilities. Class weighting is used where the model supports it.

**Required input columns ({len(FEATURES)}):** {", ".join(f"`{c}`" for c in FEATURES)}

Extra columns such as `customerID` are ignored, so the original Kaggle CSV can be uploaded directly.
""")
