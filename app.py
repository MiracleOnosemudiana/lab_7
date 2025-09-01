import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Load saved models
simple_model = joblib.load("simple_regression.pkl")
multiple_model = joblib.load("multiple_regression.pkl")

st.title("Advertising Budget vs Sales Prediction")

st.sidebar.header("Choose Model")
model_choice = st.sidebar.radio(
    "Select a model:", ("Simple Linear Regression", "Multiple Linear Regression")
)

if model_choice == "Simple Linear Regression":
    st.subheader("Simple Regression (TV → Sales)")
    tv = st.number_input("Enter TV budget ($):", min_value=0.0, step=10.0)
    if st.button("Predict Sales"):
        prediction = simple_model.predict(np.array([[tv]]))
        st.success(f"Predicted Sales: {prediction[0][0]:.2f}")

else:
    st.subheader("Multiple Regression (TV + Radio + Newspaper → Sales)")
    tv = st.number_input("Enter TV budget ($):", min_value=0.0, step=10.0)
    radio = st.number_input("Enter Radio budget ($):", min_value=0.0, step=5.0)
    newspaper = st.number_input("Enter Newspaper budget ($):", min_value=0.0, step=5.0)

    if st.button("Predict Sales"):
        features = np.array([[tv, radio, newspaper]])
        prediction = multiple_model.predict(features)
        st.success(f"Predicted Sales: {prediction[0]:.2f}")
