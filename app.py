from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "model"
TARGET_COLUMN = "y"
DATA_SEPARATOR = ";"
POSITIVE_LABEL = "yes"
REQUIRED_FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
NUMERIC_FEATURES = [
    "age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "poutcome",
]
CATEGORY_VALUES = {
    "job": {"admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"},
    "marital": {"divorced", "married", "single", "unknown"},
    "education": {"basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"},
    "default": {"no", "yes", "unknown"},
    "housing": {"no", "yes", "unknown"},
    "loan": {"no", "yes", "unknown"},
    "contact": {"cellular", "telephone"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    "day_of_week": {"mon", "tue", "wed", "thu", "fri"},
    "poutcome": {"failure", "nonexistent", "success"},
}
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}
METRIC_HELP = {
    "Accuracy": "Share of all customers classified correctly.",
    "AUC": "How well the model ranks subscribers above non-subscribers (1.0 is perfect, 0.5 is chance).",
    "Precision": "Of customers predicted to subscribe, the share who actually did.",
    "Recall": "Of customers who actually subscribed, the share the model found.",
    "F1": "Balance between precision and recall.",
    "MCC": "Single score that stays reliable when classes are imbalanced (−1 to 1).",
    "Average Precision": "Area under the precision-recall curve for the positive class.",
}

# Palette
INK = "#12302F"
TEAL = "#1F6F6B"
SAND = "#F6F4EE"
LINE = "#DDD8CB"
AMBER = "#C98A1B"
MUTED = "#5B6B69"
MODEL_COLORS = ["#1F6F6B", "#C98A1B", "#6D5BA8", "#B5493F", "#4A7FB0"]

st.set_page_config(
    page_title="Term Deposit Model Evaluation",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Public+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stDataFrame, label, button {{
        font-family: 'Public Sans', system-ui, sans-serif;
    }}
    .block-container {{ max-width: 1200px; padding-top: 1.5rem; padding-bottom: 4rem; }}
    h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif; color: {INK}; letter-spacing: -0.01em; }}
    h2 {{ font-size: 1.55rem; margin-top: 0.5rem; }}
    h3 {{ font-size: 1.2rem; }}

    .hero {{
        background: {INK}; color: #fff; border-radius: 14px;
        padding: 2rem 2.2rem; margin-bottom: 1.5rem;
    }}
    .hero h1 {{ color: #fff; font-size: 2.1rem; margin: 0 0 0.4rem 0; padding: 0; }}
    .hero p {{ color: #C9D8D6; margin: 0; max-width: 46rem; line-height: 1.55; }}

    .best-card {{
        background: #fff; border: 1px solid {LINE}; border-left: 5px solid {AMBER};
        border-radius: 10px; padding: 1.1rem 1.3rem; height: 100%;
    }}
    .best-card .label {{ color: {MUTED}; font-size: 0.85rem; }}
    .best-card .value {{ font-family: 'Fraunces', serif; font-size: 1.5rem; color: {INK}; margin: 0.15rem 0; }}
    .best-card .note {{ color: {MUTED}; font-size: 0.85rem; }}

    [data-testid="stMetric"] {{
        background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 0.9rem 1rem;
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; }}
    [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 600; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.6rem 1.1rem; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ color: {TEAL}; }}

    section[data-testid="stSidebar"] {{ background: {SAND}; border-right: 1px solid {LINE}; }}
    .step {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.6; }}
    .empty {{
        border: 1.5px dashed {LINE}; border-radius: 14px; padding: 2.5rem 2rem;
        background: #fff; text-align: center; color: {MUTED};
    }}
    .empty h3 {{ margin-top: 0; }}
    :focus-visible {{ outline: 2px solid {AMBER}; outline-offset: 2px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid", font="DejaVu Sans")


# ----------------------------------------------------------------------------
# Model loading and evaluation
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / f"{model_name}.joblib")


@st.cache_data(show_spinner=False)
def evaluate_model(model_name, evaluation_data):
    pipeline = load_model(model_name)
    features = evaluation_data.drop(columns=TARGET_COLUMN)
    target = evaluation_data[TARGET_COLUMN]
    predictions = pipeline.predict(features)

    classifier = pipeline.named_steps["classifier"]
    classes = classifier.classes_
    if not hasattr(classifier, "probability") or classifier.probability:
        probabilities = pipeline.predict_proba(features)
        scores = probabilities[:, list(classes).index(POSITIVE_LABEL)]
    else:
        scores = pipeline.decision_function(features)

    target_binary = (target == POSITIVE_LABEL).astype(int)
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target_binary, scores),
        "Precision": precision_score(target, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(target, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(target, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
        "Average Precision": average_precision_score(target_binary, scores),
    }
    return target, predictions, metrics, scores


def label_for(model_name):
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def feature_importance(model_name, evaluation_data, target):
    pipeline = load_model(model_name)
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(classifier, "coef_") or hasattr(classifier, "feature_importances_"):
        names = preprocessor.get_feature_names_out()
        values = getattr(classifier, "coef_", getattr(classifier, "feature_importances_", None))
        values = values[0] if values.ndim == 2 else values
        frame = pd.DataFrame({"Feature": names, "Importance": abs(values)})
        method = "absolute model coefficients" if hasattr(classifier, "coef_") else "tree impurity importance"
    else:
        sample = evaluation_data.sample(min(1000, len(evaluation_data)), random_state=42)
        sample_features = sample.drop(columns=TARGET_COLUMN)
        sample_target = sample[TARGET_COLUMN]
        permutation = permutation_importance(
            pipeline, sample_features, sample_target, scoring="f1", n_repeats=3,
            random_state=42, n_jobs=1,
        )
        frame = pd.DataFrame({"Feature": REQUIRED_FEATURES, "Importance": permutation.importances_mean})
        frame["Importance"] = frame["Importance"].abs()
        method = "permutation importance measured by F1 impact"
    return frame.sort_values("Importance", ascending=False).head(10), method


def business_insights(comparison_frame):
    metric_labels = {
        "F1": "balanced F1-score",
        "Recall": "recall of interested customers",
        "Precision": "precision of outreach targets",
        "AUC": "ROC-AUC discrimination",
        "MCC": "MCC under class imbalance",
    }
    statements = []
    for metric, explanation in metric_labels.items():
        winner = comparison_frame[metric].idxmax()
        statements.append(f"**{metric} leader:** {winner} has the highest {explanation} ({comparison_frame.loc[winner, metric]:.4f}).")
    return statements


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Term deposit model evaluation</h1>
        <p>Upload a held-out test file to compare five trained models on how well
        they predict which bank customers will subscribe to a term deposit.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Evaluation setup")
    st.markdown(
        '<div class="step"><b>1.</b> Upload your test CSV<br>'
        "<b>2.</b> Pick a ranking metric<br>"
        "<b>3.</b> Choose a model for details</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Test data (CSV)",
        type=["csv"],
        help=f"Semicolon-separated file with the {len(REQUIRED_FEATURES)} feature columns "
        f"and the target column '{TARGET_COLUMN}' (values: yes / no).",
    )

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty">
            <h3>Upload test data to start</h3>
            <p>Use the sidebar to add a semicolon-separated CSV that includes the
            target column <b>y</b> with <b>yes</b> / <b>no</b> values.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------------------
# Data validation
# ----------------------------------------------------------------------------
try:
    data = pd.read_csv(uploaded_file, sep=DATA_SEPARATOR)
except Exception as error:
    st.error(f"The file could not be read as a semicolon-separated CSV. Details: {error}")
    st.stop()

if TARGET_COLUMN not in data.columns:
    st.error(f"Add the target column '{TARGET_COLUMN}' to the file and upload it again.")
    st.stop()

missing_features = sorted(set(REQUIRED_FEATURES) - set(data.columns))
if missing_features:
    st.error("These required columns are missing: " + ", ".join(missing_features))
    st.stop()

invalid_labels = sorted(set(data[TARGET_COLUMN].dropna().unique()) - {"yes", "no"})
if invalid_labels or data[TARGET_COLUMN].isna().any():
    st.error(
        f"Column '{TARGET_COLUMN}' must contain only 'yes' or 'no' with no blanks. "
        f"Found: {', '.join(map(str, invalid_labels)) or 'blank values'}."
    )
    st.stop()

missing_value_count = int(data[REQUIRED_FEATURES].isna().sum().sum())
duplicate_count = int(data.duplicated().sum())
numeric_issues = {}
for column in NUMERIC_FEATURES:
    invalid_count = int(pd.to_numeric(data[column], errors="coerce").isna().sum())
    if invalid_count:
        numeric_issues[column] = invalid_count
unexpected_categories = {
    column: sorted(set(data[column].dropna().astype(str)) - allowed)
    for column, allowed in CATEGORY_VALUES.items()
    if set(data[column].dropna().astype(str)) - allowed
}

with st.sidebar:
    st.divider()
    st.subheader("Dataset status")
    st.caption(uploaded_file.name)
    st.write(f"{uploaded_file.size / 1024:.1f} KB | {len(data):,} rows | {data.shape[1]} columns")
    if missing_value_count == 0:
        st.success("No missing feature values")
    else:
        st.warning(f"{missing_value_count:,} missing feature values")
    if duplicate_count == 0:
        st.success("No duplicate records")
    else:
        st.warning(f"{duplicate_count:,} duplicate records")
    if numeric_issues:
        st.error("Invalid numeric values: " + ", ".join(numeric_issues))
    else:
        st.success("Numeric columns are valid")
    if unexpected_categories:
        st.warning("Unexpected categories will be ignored by the saved encoder")
    else:
        st.success("Categorical values are recognized")

available_models = sorted(path.stem for path in MODEL_DIR.glob("*.joblib") if path.is_file())
if not available_models:
    st.warning(
        "No trained models found. Add the five saved .joblib pipelines to the "
        "'model' folder, then reload this page."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Evaluate all models
# ----------------------------------------------------------------------------
results = {}
with st.spinner("Evaluating models. This may take a moment..."):
    for model_name in available_models:
        try:
            results[model_name] = evaluate_model(model_name, data)
        except Exception as error:
            st.error(f"{label_for(model_name)} could not be evaluated. Details: {error}")
            st.stop()

comparison = pd.DataFrame({name: res[2] for name, res in results.items()}).T
comparison.index = [label_for(name) for name in comparison.index]
comparison.index.name = "Model"

with st.sidebar:
    ranking_metric = st.selectbox(
        "Rank models by",
        list(comparison.columns),
        index=list(comparison.columns).index("F1"),
        help=METRIC_HELP["F1"] + " Changing this updates the ranking and the best-model card.",
    )
    selected_model = st.selectbox(
        "Model for detailed results",
        available_models,
        format_func=label_for,
    )
    st.divider()
    st.download_button(
        "Download comparison (CSV)",
        comparison.round(4).to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

ranked = comparison.sort_values(ranking_metric, ascending=False)
best_name = ranked.index[0]
best_value = ranked.iloc[0][ranking_metric]

# ----------------------------------------------------------------------------
# Summary row
# ----------------------------------------------------------------------------
positive_rate = (data[TARGET_COLUMN] == POSITIVE_LABEL).mean()
col_best, col_rows, col_rate, col_models = st.columns([1.6, 1, 1, 1], gap="medium")
with col_best:
    st.markdown(
        f"""
        <div class="best-card">
            <div class="label">Top model by {ranking_metric}</div>
            <div class="value">{best_name}</div>
            <div class="note">{ranking_metric} of {best_value:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
col_rows.metric("Customers evaluated", f"{len(data):,}")
col_rate.metric("Subscribed (yes)", f"{positive_rate:.1%}")
col_models.metric("Models compared", len(results))

st.write("")

tab_overview, tab_detail, tab_data, tab_compare = st.tabs(
    ["Overview", "Model details", "Dataset analysis", "Model comparison"]
)

# ----------------------------------------------------------------------------
# Tab 1: overview
# ----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Business insights & model recommendations")
    st.caption("These statements are calculated from the uploaded test results.")
    for statement in business_insights(comparison):
        st.markdown(statement)
    st.info(
        "Prioritize **Recall** when missing a potential subscriber is costly, "
        "**Precision** when unnecessary outreach is costly, **F1** for a balance, "
        "and **ROC-AUC** for ranking quality across thresholds. No model is "
        "universally best for every campaign objective."
    )

    st.subheader("Positive-class distribution")
    distribution = data[TARGET_COLUMN].value_counts().reindex(["no", "yes"], fill_value=0)
    distribution_frame = pd.DataFrame({"Outcome": distribution.index, "Customers": distribution.values})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.barplot(
        data=distribution_frame,
        x="Outcome",
        y="Customers",
        hue="Outcome",
        palette=[TEAL, AMBER],
        legend=False,
        ax=ax,
    )
    for index, value in enumerate(distribution.values):
        ax.text(index, value, f"{value:,}\n{value / len(data):.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    sns.despine(fig)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ----------------------------------------------------------------------------
# Tab 4: comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Scores at a glance")
    st.caption(f"Sorted by {ranking_metric}. The best score in each column is highlighted.")
    styled = (
        ranked.style.format("{:.4f}")
        .highlight_max(axis=0, color="#F3E3BF")
    )
    st.dataframe(styled, width="stretch")

    st.subheader("Metric comparison")
    st.caption("All scores are calculated on the uploaded held-out test data.")
    melted = ranked.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.barplot(
        data=melted, x="Metric", y="Score", hue="Model",
        palette=MODEL_COLORS[: len(ranked)], ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylim(min(0, melted["Score"].min() - 0.05), 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    sns.despine(fig)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.28)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    
    st.subheader("ROC curves")
    st.caption(
        "The dashed diagonal represents a random classifier. "
        "AUC values are shown in the legend."
    )

    fig, ax = plt.subplots(figsize=(11, 5.8))

    for color, (name, res) in zip(MODEL_COLORS, results.items()):
        fpr, tpr, _ = roc_curve(
            (res[0] == POSITIVE_LABEL).astype(int),
            res[3]
        )

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{label_for(name)} ({res[2]['AUC']:.3f})"
        )

    # Random classifier reference
    ax.plot(
        [0, 1],
        [0, 1],
        ls="--",
        lw=1.2,
        color="#9AA5A3",
        label="Random classifier"
    )

    # Axis labels
    ax.set_xlabel("False positive rate", labelpad=12)
    ax.set_ylabel("True positive rate", labelpad=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Legend with sufficient separation
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        title="Model (AUC)",
        title_fontsize=9
    )

    sns.despine(fig)

    # Reserve space for the legend
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.34
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


    with st.expander("What do these metrics mean?"):
        for metric, text in METRIC_HELP.items():
            st.markdown(f"**{metric}** — {text}")

# ----------------------------------------------------------------------------
# Tab 2: detailed view
# ----------------------------------------------------------------------------
with tab_detail:
    target, predictions, selected_metrics, _ = results[selected_model]
    st.subheader(label_for(selected_model))
    st.caption("Change the model from the sidebar.")

    metric_items = list(selected_metrics.items())
    for row_start in (0, 3, 6):
        cols = st.columns(3, gap="medium")
        for col, (name, value) in zip(cols, metric_items[row_start: row_start + 3]):
            col.metric(name, f"{value:.4f}", help=METRIC_HELP.get(name))

    st.write("")
    matrix_col, report_col = st.columns([0.9, 1.1], gap="large")

    with matrix_col:
        st.markdown("##### Confusion matrix")
        st.caption("Rows are actual outcomes, columns are predictions. Cells show counts and row share.")
        matrix = confusion_matrix(target, predictions, labels=["no", "yes"])
        row_share = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        annotations = np.array(
            [[f"{matrix[i, j]:,}\n{row_share[i, j]:.1%}" for j in range(2)] for i in range(2)]
        )
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(
            matrix, annot=annotations, fmt="", cmap=sns.light_palette(TEAL, as_cmap=True),
            cbar=False, square=True, linewidths=2, linecolor="white",
            xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax,
            annot_kws={"fontsize": 10},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig, width="content")
        plt.close(fig)

    with report_col:
        st.markdown("##### Classification report")
        st.caption("Per-class precision, recall and F1.")
        report = classification_report(
            target, predictions, labels=["no", "yes"], target_names=["no", "yes"],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                    "f1-score": "{:.4f}", "support": "{:,}"}),
            width="stretch",
        )

    st.markdown("##### Model interpretability")
    importance, importance_method = feature_importance(selected_model, data, target)
    st.caption(
        f"Top ten features by {importance_method}. Importance indicates association "
        "with predictions, not causation."
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    plot_data = importance.sort_values("Importance", ascending=True)
    ax.barh(plot_data["Feature"], plot_data["Importance"], color=TEAL)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    sns.despine(fig)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("##### Prediction analysis")
    target, predictions, _, scores = results[selected_model]
    prediction_sample = data.drop(columns=TARGET_COLUMN).head(100).copy()
    prediction_sample.insert(0, "Actual", target.head(100).to_numpy())
    prediction_sample.insert(1, "Predicted", predictions[:100])
    prediction_sample.insert(2, "Positive-class score", scores[:100])
    prediction_sample.insert(
        3,
        "Result",
        np.where(predictions[:100] == target.head(100).to_numpy(), "Correct", "Incorrect"),
    )
    st.caption("The score is a probability for probabilistic models and a decision score for SVM.")
    st.dataframe(prediction_sample, width="stretch", height=300)

# ----------------------------------------------------------------------------
# Tab 3: data preview
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Uploaded data")
    st.caption(f"{len(data):,} rows and {data.shape[1]} columns. Showing the first 100 rows.")
    st.dataframe(data.head(100), width="stretch", height=420)
    data_col, validation_col = st.columns(2, gap="large")
    with data_col:
        st.markdown("##### Dataset profile")
        st.write(f"Numerical features: {len(NUMERIC_FEATURES)}")
        st.write(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
        st.write(f"Missing values: {missing_value_count:,}")
        st.write(f"Duplicate records: {duplicate_count:,}")
        st.write("Feature columns: " + ", ".join(REQUIRED_FEATURES))
    with validation_col:
        st.markdown("##### Validation summary")
        validation_summary = pd.DataFrame(
            {
                "Check": ["Required columns", "Target labels", "Missing values", "Duplicate records", "Numeric values", "Categorical values"],
                "Status": [
                    "Pass",
                    "Pass",
                    "Warning" if missing_value_count else "Pass",
                    "Warning" if duplicate_count else "Pass",
                    "Warning" if numeric_issues else "Pass",
                    "Warning" if unexpected_categories else "Pass",
                ],
            }
        )
        st.dataframe(validation_summary, hide_index=True, width="stretch")
        st.download_button(
            "Download validation summary",
            validation_summary.to_csv(index=False).encode("utf-8"),
            file_name="dataset_validation_summary.csv",
            mime="text/csv",
        )