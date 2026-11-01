from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "model"
TARGET_COLUMN = "y"
DATA_SEPARATOR = ";"
REQUIRED_FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}


st.set_page_config(
    page_title="Term Deposit Prediction",
    layout="wide",
)

st.title("Bank Term Deposit Prediction")
st.write(
    "Compare five saved classifiers on the uploaded bank marketing test data. "
    "The app loads trained pipelines and never retrains on uploaded data."
)

uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload test_data.csv to begin.")
    st.stop()

data = pd.read_csv(uploaded_file, sep=DATA_SEPARATOR)
st.subheader("Uploaded data")
st.dataframe(data.head(10), use_container_width=True)

if TARGET_COLUMN not in data.columns:
    st.error(f"The uploaded file must contain the target column '{TARGET_COLUMN}'.")
    st.stop()

missing_features = sorted(set(REQUIRED_FEATURES) - set(data.columns))
if missing_features:
    st.error("The uploaded file is missing required features: " + ", ".join(missing_features))
    st.stop()

available_models = sorted(
    path.stem for path in MODEL_DIR.glob("*.joblib") if path.is_file()
)

if not available_models:
    st.warning(
        "No trained model files are available yet. Add the five saved .joblib "
        "pipelines to the model folder before evaluating predictions."
    )
    st.stop()

selected_model = st.selectbox(
    "Select a model for detailed analysis",
    available_models,
    format_func=lambda model_name: MODEL_LABELS.get(model_name, model_name),
)


def evaluate_model(model_name):
    pipeline = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    predictions = pipeline.predict(features)
    classes = pipeline.named_steps["classifier"].classes_
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "probability") or classifier.probability:
        probabilities = pipeline.predict_proba(features)
        positive_index = list(classes).index("yes")
        positive_scores = probabilities[:, positive_index]
    else:
        positive_scores = pipeline.decision_function(features)
    target_binary = (target == "yes").astype(int)
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target_binary, positive_scores),
        "Precision": precision_score(target, predictions, pos_label="yes", zero_division=0),
        "Recall": recall_score(target, predictions, pos_label="yes", zero_division=0),
        "F1": f1_score(target, predictions, pos_label="yes", zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return target, predictions, metrics


results = {}
for model_name in available_models:
    try:
        results[model_name] = evaluate_model(model_name)
    except Exception as error:
        st.error(f"Could not evaluate {model_name}: {error}")
        st.stop()

st.subheader("Model comparison")
comparison = pd.DataFrame(
    {model_name: result[2] for model_name, result in results.items()}
).T
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

target, predictions, selected_metrics = results[selected_model]
st.subheader(f"Detailed results: {selected_model}")
metric_columns = st.columns(6)
for column, (metric_name, metric_value) in zip(metric_columns, selected_metrics.items()):
    column.metric(metric_name, f"{metric_value:.4f}")

st.subheader("Confusion matrix")
matrix = confusion_matrix(target, predictions, labels=["no", "yes"])
figure, axis = plt.subplots(figsize=(5, 4))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["no", "yes"],
    yticklabels=["no", "yes"],
    ax=axis,
)
axis.set_xlabel("Predicted label")
axis.set_ylabel("True label")
st.pyplot(figure)
plt.close(figure)

st.subheader("Classification report")
report = classification_report(
    target,
    predictions,
    labels=["no", "yes"],
    target_names=["no", "yes"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)