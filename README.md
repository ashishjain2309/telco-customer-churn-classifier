# Telco Customer Churn Prediction

Predicting which telecom customers will churn with five machine learning models and a Streamlit app.

**GitHub repository:** https://github.com/ashishjain2309/telco-customer-churn-classifier

**Live app:** _link added after deployment_

## a. Problem statement

A telecom operator loses revenue every time a subscriber leaves (churns). Retention offers work best when they reach the right customers before they leave, so the goal is to predict, from a customer's contract, services and billing details, whether that customer will churn. This is a binary classification problem with **Churn = Yes** as the positive class. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) were trained on the same stratified train/test split and compared on six metrics.

## b. Dataset description

| Item | Value |
|---|---|
| Source | [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), IBM sample data (the raw dataset is not included in this repository) |
| Instances | 7,043 customers, one row each; no duplicate customer IDs |
| Features | 19 predictive columns: 16 categorical (for example `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `SeniorCitizen`) and 3 numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) |
| Target | `Churn`: Yes = 1 (churned), No = 0 |
| Class counts | No 5,174 (73.46%), Yes 1,869 (26.54%); imbalance ratio 2.77 |

Preprocessing:

- `customerID` was dropped: it is an identifier, not a predictor.
- `TotalCharges` is stored as text; converting it to numbers left 11 blanks, all for customers with `tenure` = 0 who had never been billed, so they were set to 0.
- The target was encoded as Yes = 1, No = 0.
- 40 rows are identical to another row on all 19 features but belong to different customers, so they were kept. In 18 of these groups the customers have different churn outcomes, which shows the features cannot fully explain churn.
- The data was split 80/20 with stratification (`random_state = 42`): 5,634 training rows and 1,409 test rows, both with 26.54% churn. The test rows are saved as `models/test_set.csv`.
- The 16 categorical columns are one-hot encoded (`OneHotEncoder`, two-value columns as a single 0/1 column), giving 40 model inputs. The 3 numeric columns use `StandardScaler` for Logistic Regression, KNN and SVM, `PowerTransformer` (Yeo–Johnson) for Gaussian Naive Bayes, and raw values for the Decision Tree. All encoding and scaling is part of each saved pipeline and was fitted on the training data only.
- `class_weight="balanced"` is used for Logistic Regression, Decision Tree and SVM.

Precision, Recall and F1 are calculated for the **Churn** class, the customers the business wants to find. AUC is computed from the predicted churn probability, and MCC summarises the whole 2 × 2 confusion matrix. Accuracy alone is misleading here: predicting "No churn" for everyone already scores 73.46%.

## c. GitHub repository link

https://github.com/ashishjain2309/telco-customer-churn-classifier

Files in the repository:

- `app.py`: the Streamlit app, which loads the saved models and never retrains
- `requirements.txt`: dependencies, pinned to scikit-learn 1.8.0 (the training version)
- `models/`: the five saved `.joblib` pipelines (encoding + scaling + model), `metadata.json` (feature order, column types, class names, training settings), `test_set.csv` (the 1,409 held-out customers with labels) and `metrics_project_b.csv` (the metrics table below)
- `notebooks/project_b_notebook.ipynb`: EDA, preprocessing, training, evaluation, saving and validation of the saved files

To run the app locally: `pip install -r requirements.txt`, then `streamlit run app.py`.

## d. Models used

Results on the 1,409 held-out test customers (374 churners):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.7381 | 0.8415 | 0.5043 | 0.7834 | 0.6136 | 0.4531 |
| Decision Tree | 0.7360 | 0.6566 | 0.5028 | 0.4866 | 0.4946 | 0.3160 |
| KNN | 0.7622 | 0.7766 | 0.5543 | 0.5321 | 0.5430 | 0.3825 |
| Naive Bayes | 0.6849 | 0.8128 | 0.4499 | 0.8396 | 0.5858 | 0.4138 |
| Support Vector Machine | 0.7821 | 0.8219 | 0.5827 | 0.6310 | 0.6059 | 0.4564 |

Settings: Logistic Regression `max_iter=5000`, Decision Tree with default depth (unpruned), KNN `k=5`, Gaussian Naive Bayes with defaults after a Yeo–Johnson power transform of the numeric columns, SVM with RBF kernel and default `C=1`, with probabilities calibrated by `CalibratedClassifierCV` (sigmoid, 5-fold).

Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The best AUC (0.8415) and F1 (0.6136), and an MCC (0.4531) almost equal to the SVM. Class weighting makes it catch 293 of the 374 churners (recall 0.7834), but it flags 581 customers, 288 of whom do not churn (precision 0.5043). Its coefficients are easy to explain: longer `tenure` (−1.14) and a two-year contract (−0.81) are the strongest signs a customer will stay, while fibre-optic internet (+0.67) and a month-to-month contract (+0.62) push towards churn. |
| Decision Tree | The weakest model: AUC 0.6566, F1 0.4946 and MCC 0.3160. It finds only 182 of the 374 churners (recall 0.4866), and its accuracy (0.7360) is barely above the 73.46% of always predicting "No churn". The unpruned tree grew to depth 26 with 1,250 leaves for 5,634 training rows, so it memorises individual customers, and its leaves give almost only 0/1 probabilities, which hurts AUC. |
| KNN | Accuracy 0.7622 but recall only 0.5321 (199 of 374 churners) and MCC 0.3825. KNN has no class weighting, so the majority "No churn" class dominates the neighbourhood votes. Distances over 40 mostly one-hot inputs are also a weak notion of similarity. The saved model stores all 5,634 training rows. |
| Naive Bayes | The highest recall (0.8396, 314 of 374 churners), but it flags 698 customers, 384 of whom do not churn, giving the lowest precision (0.4499) and accuracy (0.6849). The one-hot 0/1 inputs do not follow a bell curve, and the "No internet service" value repeated across six columns breaks the independence assumption. Its AUC (0.8128) is still reasonable, so its ranking of customers is better than its yes/no decisions. |
| Support Vector Machine | The best Accuracy (0.7821), Precision (0.5827) and MCC (0.4564): 1,102 of 1,409 customers correct. It finds 236 of the 374 churners (recall 0.6310) with 169 false alarms, the most balanced trade-off of the five models. Its AUC (0.8219) and F1 (0.6059) are second only to Logistic Regression. |
| Overall winner | Support Vector Machine. It has the best Accuracy, Precision and MCC and is second on AUC and F1. Logistic Regression is a very close second (MCC 0.4531 against 0.4564) with the best AUC and F1 and much higher recall. The better choice depends on the cost of a retention action: for cheap offers (for example a discount SMS) Logistic Regression catches more churners; for expensive ones (for example a call from a retention agent) SVM wastes fewer contacts. All models score lower than on cleaner datasets because churn depends on factors the data does not record, as the 18 groups of identical customers with different outcomes show. |

Limitation: for churned customers, `tenure` is their full lifetime with the operator, while for current customers it is their tenure so far, so part of the strong tenure effect comes from how this snapshot was taken.

## e. Streamlit app

- **Sidebar:** pick a model from the dropdown, upload a CSV or tick **Use the bundled test set**, and download the bundled `test_set.csv`.
- **Predict & evaluate tab:** the six metrics for the chosen model, a 2 × 2 confusion matrix, a classification report, and a predictions table with the churn probability that can be downloaded as CSV.
- **Model comparison tab:** the metrics table for all five models, with the best value in each column highlighted.
- **About the data tab:** dataset, target, preprocessing and the list of required input columns.

The uploaded CSV needs the 19 customer columns. Extra columns (such as `customerID`) and a different column order are fine, so the original Kaggle CSV can be uploaded directly. The app applies the same cleaning as training: blank `TotalCharges` values become 0 (with a message), `SeniorCitizen` may be 0/1 or Yes/No, and `Churn` labels may be Yes/No or 1/0. Category values never seen in training are reported. A missing column, an empty file or a file that is not a CSV shows an error message; rows with non-numeric `tenure` or charges are skipped with a warning. If the labels contain only one class, AUC shows n/a. The app only loads the saved models from `models/`; it never trains.
