# streamlit_chd_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from io import BytesIO

st.set_page_config(page_title="10-Year CHD Risk Predictor", layout="centered")
MODEL_DIR = "saved_chd_model"

@st.cache_data
def load_metadata():
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        return json.load(f)

@st.cache_resource
def load_artifacts():
    meta = load_metadata()
    scaler = joblib.load(os.path.join(MODEL_DIR, meta["scaler_file"]))
    model = joblib.load(os.path.join(MODEL_DIR, meta["model_file"]))
    return meta, scaler, model

try:
    meta, scaler, model = load_artifacts()
    FEATURE_NAMES = meta["feature_names"]
except Exception as e:
    st.error(f"Could not load model artifacts from '{MODEL_DIR}': {e}")
    st.stop()

st.title("🔎 10-Year CHD Risk Predictor (Logistic Regression)")
st.markdown(
    "Provide patient features and get a prediction whether they are likely to develop CHD within 10 years.\n\n"
    "**Model input features**: " + ", ".join(FEATURE_NAMES)
)

mode = st.sidebar.radio("Mode", ["Single input form", "Batch CSV upload"])

def prepare_input_df(df):
    # keep only required features and in the required order
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[FEATURE_NAMES].astype(float).values
    X_scaled = scaler.transform(X)
    return X_scaled

def predict_proba_and_label(X_scaled):
    # logistic regression supports predict_proba
    probs = model.predict_proba(X_scaled)[:, 1]  # probability of class 1 (TenYearCHD)
    preds = model.predict(X_scaled).astype(int)
    return preds, probs

if mode == "Single input form":
    st.subheader("Single patient input")
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Age (years)", min_value=0.0, max_value=120.0, value=50.0, step=1.0, format="%.0f")
        sex_male = col1.selectbox("Sex (male = 1, female = 0)", options=[1, 0], index=1)
        cigsPerDay = col2.number_input("Cigarettes per day", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
        totChol = col1.number_input("Total cholesterol (mg/dL)", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)
        sysBP = col2.number_input("Systolic BP (mm Hg)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        glucose = col2.number_input("Glucose (mg/dL)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        try:
            df_in = pd.DataFrame([[age, sex_male, cigsPerDay, totChol, sysBP, glucose]], columns=FEATURE_NAMES)
            Xs = prepare_input_df(df_in)
            preds, probs = predict_proba_and_label(Xs)
            label = "TenYearCHD" if int(preds[0]) == 1 else "No TenYearCHD"
            st.metric("Prediction", label, delta=f"Risk probability: {probs[0]:.4f}")
            st.write(f"Probability of 10-year CHD: **{probs[0]:.4f}**")
            st.write("Note: This is a model prediction — use clinical judgement and additional tests.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Batch CSV upload")
    st.markdown("Upload a CSV containing the columns: " + ", ".join(FEATURE_NAMES))
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None
        if df is not None:
            try:
                Xs = prepare_input_df(df)
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                st.stop()
            preds, probs = predict_proba_and_label(Xs)
            out = df.copy()
            out["prediction"] = preds
            out["risk_probability"] = probs
            st.success(f"Predicted {len(out)} rows")
            st.dataframe(out.head(100))

            # provide CSV download
            towrite = BytesIO()
            out.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button("Download predictions CSV", data=towrite, file_name="chd_predictions.csv", mime="text/csv")

st.caption("Model and scaler loaded from 'saved_chd_model'. The app applies the same StandardScaler as used during training.")
