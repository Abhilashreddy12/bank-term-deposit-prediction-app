# Bank Term Deposit Prediction

## Problem statement

This project predicts whether a bank customer will subscribe to a term deposit
after a marketing campaign. Five classification models are trained on the same
dataset, compared using six evaluation metrics, and served through Streamlit.

## Dataset description

The project uses the UCI Bank Marketing dataset with social and economic
context. It contains 41,188 records, 20 input features, and the binary target
column `y` (`yes` or `no`). The files are semicolon-delimited.

`training_data/bank-additional-train.csv` contains 32,940 stratified training
records. `test_data.csv` contains 8,236 held-out stratified test records. The
split uses `random_state=42`, and repeated source rows were removed from the
generated split so that identical rows do not cross the train/test boundary.

The target is naturally imbalanced. `RandomOverSampler` is applied inside each
training pipeline only; the test data is never resampled. The `duration`
feature is included for this academic benchmark, but should be excluded from a
real pre-call production system because it is known only after the call.

## GitHub repository link

[https://github.com/Abhilashreddy12/bank-term-deposit-prediction-app](https://github.com/Abhilashreddy12/bank-term-deposit-prediction-app)

## Models used and evaluation metrics

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

The training script calculates accuracy, AUC, precision, recall, F1 score, and
Matthews correlation coefficient (MCC). It writes final test results to
`metrics.csv` and validation results to `validation_metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8641 | 0.9402 | 0.4492 | 0.9095 | 0.6014 | 0.5786 |
| Decision Tree | 0.8610 | 0.8951 | 0.4402 | 0.8610 | 0.5826 | 0.5507 |
| KNN | 0.8418 | 0.9013 | 0.4031 | 0.8405 | 0.5449 | 0.5090 |
| Naive Bayes | 0.8004 | 0.8394 | 0.3287 | 0.7403 | 0.4553 | 0.3984 |
| Support Vector Machine | 0.8613 | 0.9387 | 0.4443 | 0.9203 | 0.5993 | 0.5787 |

## Model observations

Complete these observations from the generated `metrics.csv`:

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strongest AUC (0.9402), high recall (0.9095), and the best F1 score (0.6014). It provides a strong interpretable baseline. |
| Decision Tree | Good accuracy (0.8610), but lower AUC and MCC than the linear and kernel models. The depth and leaf constraints help reduce overfitting. |
| KNN | Scaling and distance weighting produce reasonable recall, but its AUC, F1, and MCC are lower than the other leading models. |
| Naive Bayes | Fastest and simplest probabilistic model, but its independence assumption gives the weakest overall results on this dataset. |
| Support Vector Machine | Highest recall (0.9203) and MCC (0.5787), with performance close to Logistic Regression, but it is more computationally expensive. |
| Overall winner | Logistic Regression is the overall winner because it has the best AUC and F1 while maintaining strong recall. SVM is preferable when maximizing minority-class recall is the priority. |

## Project structure

```text
bank-term-deposit-prediction-app/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- metrics.csv
|-- validation_metrics.csv
|-- training_data/
|   `-- bank-additional-train.csv
`-- model/
	|-- logistic_regression.joblib
	|-- decision_tree.joblib
	|-- knn.joblib
	|-- naive_bayes.joblib
	`-- svm.joblib
```

## Training the models

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_models.py
```

The script saves complete preprocessing, balancing, and classifier pipelines
in `model/`. It does not use or modify `test_data.csv` during training.

## Running the Streamlit application

```powershell
streamlit run app.py
```

Upload `test_data.csv`. The app loads the saved models and displays results for
all five models, including accuracy, AUC, precision, recall, F1, MCC, a
confusion matrix, and a classification report. The app never retrains on
uploaded data.

## Streamlit Community Cloud link

Add the deployed Streamlit URL here before submission.
