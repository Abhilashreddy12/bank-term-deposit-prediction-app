# Bank Term Deposit Prediction

## Problem statement

This project compares five supervised classification models for predicting
whether a bank customer will subscribe to a term deposit. The Streamlit app
will allow evaluation of saved models using a test CSV without retraining on
uploaded data.

## Dataset description

The project uses the UCI Bank Marketing dataset. It contains 41,188 customer
records and 20 input features covering customer information, campaign contact
details, and economic indicators. The target column is `y`:

- `yes`: the customer subscribed to a term deposit
- `no`: the customer did not subscribe

The source file is semicolon-delimited. The dataset contains more than the
required 12 features and 500 instances.

## Project structure

```text
bank-term-deposit-prediction-app/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
`-- model/
```

## Models to be implemented

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

Each model will be evaluated using accuracy, AUC, precision, recall, F1
score, and Matthews correlation coefficient.

## Running the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the application. Trained model pipelines will be
loaded from the `model/` directory once they have been created.

## Repository link

Add the GitHub repository URL here after the repository is published.

## Streamlit link

Add the deployed Streamlit Community Cloud URL here after deployment.
