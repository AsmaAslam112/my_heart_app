
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "heart_svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


st.title("Heart Disease Prediction System")
st.write("Enter the patient's details to predict the model's result.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Please upload heart_svc_trained_model.pkl to this repository.")
    st.stop()

with st.form("prediction_form"):
    age = st.number_input("Age", 1, 120, 55)

    sex = st.selectbox(
        "Sex",
        [1, 0],
        format_func=lambda x: "Male (1)" if x == 1 else "Female (0)"
    )

    cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)", 40, 250, 150
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    prediction = int(model.predict(patient)[0])

    if prediction == 1:
        st.warning("Prediction: Heart disease may be present.")
    else:
        st.success("Prediction: Heart disease not detected by the model.")

    st.caption(
        "For educational purposes only. This prediction is not a medical diagnosis."
    )
