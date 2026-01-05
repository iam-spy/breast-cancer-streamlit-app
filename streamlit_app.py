import streamlit as st
import joblib
import numpy as np

st.title("Breast Cancer Prediction Web App")

# Load model
model = joblib.load("cancer_model.pkl")

st.write("Enter patient feature values below:")

def user_input():
    radius_mean = st.number_input("radius_mean")
    texture_mean = st.number_input("texture_mean")
    perimeter_mean = st.number_input("perimeter_mean")
    area_mean = st.number_input("area_mean")
    smoothness_mean = st.number_input("smoothness_mean")
    compactness_mean = st.number_input("compactness_mean")
    concavity_mean = st.number_input("concavity_mean")
    concave_points_mean = st.number_input("concave_points_mean")
    symmetry_mean = st.number_input("symmetry_mean")
    fractal_dimension_mean = st.number_input("fractal_dimension_mean")

    values = [
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean,
        concave_points_mean, symmetry_mean, fractal_dimension_mean
    ]
    return np.array(values).reshape(1, -1)

data = user_input()

if st.button("Predict"):
    pred = model.predict(data)[0]
    result = "Malignant" if pred == "M" else "Benign"

    st.success(f"Prediction Result: {result}")
