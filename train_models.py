from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).parent
TRAIN_PATH = PROJECT_ROOT / "training_data" / "bank-additional-train.csv"
TEST_PATH = PROJECT_ROOT / "test_data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TARGET_COLUMN = "y"
RANDOM_STATE = 42

MODEL_SPECS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, random_state=RANDOM_STATE),
    ),
    "knn": ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=15, weights="distance")),
    "naive_bayes": ("Gaussian Naive Bayes", GaussianNB()),
    "svm": (
        "Support Vector Machine",
        SVC(C=1.0, kernel="rbf", probability=False, random_state=RANDOM_STATE),
    ),
}


def build_pipeline(model):
    numeric_columns = [
        "age", "duration", "campaign", "pdays", "previous",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    ]
    categorical_columns = [
        "job", "marital", "education", "default", "housing", "loan", "contact",
        "month", "day_of_week", "poutcome",
    ]
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def calculate_metrics(pipeline, features, target):
    predictions = pipeline.predict(features)
    classes = pipeline.named_steps["classifier"].classes_
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "probability") or classifier.probability:
        probabilities = pipeline.predict_proba(features)
        positive_index = list(classes).index("yes")
        positive_scores = probabilities[:, positive_index]
    else:
        positive_scores = pipeline.decision_function(features)
    return {
        "accuracy": accuracy_score(target, predictions),
        "auc": roc_auc_score((target == "yes").astype(int), positive_scores),
        "precision": precision_score(target, predictions, pos_label="yes", zero_division=0),
        "recall": recall_score(target, predictions, pos_label="yes", zero_division=0),
        "f1": f1_score(target, predictions, pos_label="yes", zero_division=0),
        "mcc": matthews_corrcoef(target, predictions),
    }


def main():
    training_data = pd.read_csv(TRAIN_PATH, sep=";")
    test_data = pd.read_csv(TEST_PATH, sep=";")
    features = training_data.drop(columns=TARGET_COLUMN)
    target = training_data[TARGET_COLUMN]
    test_features = test_data.drop(columns=TARGET_COLUMN)
    test_target = test_data[TARGET_COLUMN]
    train_features, validation_features, train_target, validation_target = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )

    MODEL_DIR.mkdir(exist_ok=True)
    validation_rows = []
    test_rows = []
    for model_key, (display_name, model) in MODEL_SPECS.items():
        validation_pipeline = build_pipeline(model)
        validation_pipeline.fit(train_features, train_target)
        validation_rows.append(
            {"model": display_name, "model_key": model_key,
             **calculate_metrics(validation_pipeline, validation_features, validation_target)}
        )
        final_pipeline = build_pipeline(model)
        final_pipeline.fit(features, target)
        joblib.dump(final_pipeline, MODEL_DIR / f"{model_key}.joblib")
        test_rows.append(
            {"model": display_name, "model_key": model_key,
             **calculate_metrics(final_pipeline, test_features, test_target)}
        )
        print(f"Saved {display_name}")

    pd.DataFrame(validation_rows).to_csv(PROJECT_ROOT / "validation_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(PROJECT_ROOT / "metrics.csv", index=False)
    print("Saved validation_metrics.csv and metrics.csv")


if __name__ == "__main__":
    main()