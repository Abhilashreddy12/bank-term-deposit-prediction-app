from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "model"
TARGET_COLUMN = "y"
DATA_SEPARATOR = ";"


st.set_page_config(
    page_title="Term Deposit Prediction",
    layout="wide",
)

st.title("Bank Term Deposit Prediction")
st.write(
    "Upload the semicolon-delimited test data to inspect the bank marketing "
    "classification experiment."
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

available_models = sorted(
    path.stem for path in MODEL_DIR.glob("*.joblib") if path.is_file()
)

if not available_models:
    st.warning(
        "No trained model files are available yet. Add the five saved .joblib "
        "pipelines to the model folder before evaluating predictions."
    )
    st.stop()

selected_model = st.selectbox("Select a model", available_models)
st.caption(f"Selected model: {selected_model}")