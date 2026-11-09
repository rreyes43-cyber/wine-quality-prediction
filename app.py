import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Prediction System")
st.caption("A classroom prototype for preliminary red-wine quality screening.")

required_files = ["wine_model.pkl", "wine_quality.csv"]
missing_files = [name for name in required_files if not Path(name).exists()]
if missing_files:
    st.error("Required project files are missing: " + ", ".join(missing_files))
    st.info("Run `python train_model.py` first. It downloads the dataset and creates the trained model.")
    st.stop()

model = load_model()
data = pd.read_csv("wine_quality.csv")

c1, c2, c3 = st.columns(3)
c1.metric("Dataset records", f"{len(data):,}")
c2.metric("Average quality score", f"{data['quality'].mean():.2f}")
c3.metric("Good-quality records", f"{int((data['quality'] >= 6).sum()):,}")

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV containing the 11 physicochemical measurement columns. Do not include the quality target.")
feature_columns = list(model.feature_names_in_)
template = data[feature_columns].head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [col for col in feature_columns if col not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            # Ignore extra columns; use the exact training feature order.
            predictions = model.predict(new_data[feature_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            good_count = int((predictions == 1).sum())
            st.metric("Predicted good-quality samples", good_count)
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "wine_quality_predictions.csv",
                "text/csv"
            )
    except Exception as exc:
        st.error(f"Could not process this CSV file: {exc}")

st.warning(
    "Responsible use: This is a classroom prototype trained on a particular public "
    "red-wine dataset. Predictions are preliminary and are not a substitute for "
    "laboratory testing, sensory evaluation, food-safety procedures, or professional decisions."
)
