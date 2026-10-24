"""Diamond Price Prediction – Streamlit web app.
Hamna Mubarak (FA24-BSE-046, Section A) – ML Fundamentals, Assignment 3
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("diamond_price_model.joblib")


model = load_model()

st.title("💎 Diamond Price Predictor")
st.caption("Hamna Mubarak · FA24-BSE-046 · Section A — Extra Trees model selected by 10-fold cross validation")

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat (weight)", 0.2, 5.01, 1.01, 0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color (D = best, J = worst)", list("DEFGHIJ"), index=3)
    clarity = st.selectbox("Clarity (IF = best, I1 = worst)",
                           ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"], index=4)
    depth = st.number_input("Depth (%)", 43.0, 79.0, 61.8, 0.1)
with col2:
    table = st.number_input("Table (%)", 43.0, 95.0, 57.0, 0.5)
    x = st.number_input("Length x (mm)", 3.0, 11.0, 6.42, 0.01)
    y = st.number_input("Width y (mm)", 3.0, 11.0, 6.46, 0.01)
    z = st.number_input("Depth z (mm)", 1.0, 7.0, 3.98, 0.01)

if st.button("Predict Price", type="primary", use_container_width=True):
    features = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
        "cut": cut, "color": color, "clarity": clarity,
    }])
    price = model.predict(features)[0]
    st.success(f"Predicted price: **${price:,.2f}**")
    st.caption("Typical error (10-fold CV RMSE) is about $530.")
