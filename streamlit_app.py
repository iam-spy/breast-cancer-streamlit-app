import streamlit as st
import joblib
import numpy as np

st.title("Breast Cancer Prediction Web App")

# Load trained model
model = joblib.load("cancer_model.pkl")

st.write("Enter patient feature values below or auto-fill sample values.")

# -------- Feature List (All 30) -------- #
feature_names = [
 'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
 'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',
 'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
 'smoothness_se','compactness_se','concavity_se','concave points_se',
 'symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
 'perimeter_worst','area_worst','smoothness_worst','compactness_worst',
 'concavity_worst','concave points_worst','symmetry_worst',
 'fractal_dimension_worst'
]

# -------- Auto-Fill Sample Inputs -------- #

benign_sample = [
  12.5,14.2,85,520,0.09,0.08,0.03,0.02,0.18,0.06,
  0.25,0.90,2.1,25,0.007,0.020,0.030,0.015,0.19,0.065,
  14.8,17.9,98,690,0.11,0.12,0.10,0.05,0.23,0.07
]

malignant_sample = [
  20.3,25.1,135,1250,0.12,0.20,0.25,0.12,0.25,0.08,
  0.60,1.50,4.5,80,0.010,0.045,0.120,0.060,0.30,0.090,
  25.4,32.5,178,2100,0.15,0.28,0.35,0.18,0.31,0.10
]

# maintain state for auto filling
if "values" not in st.session_state:
    st.session_state.values = [0.0] * 30

# Buttons to auto-fill values
col1, col2 = st.columns(2)

with col1:
    if st.button("Auto Fill — Benign Sample"):
        st.session_state.values = benign_sample

with col2:
    if st.button("Auto Fill — Malignant Sample"):
        st.session_state.values = malignant_sample


# -------- Input Controls -------- #
inputs = []
st.subheader("Enter / Review Feature Values")

for i, feature in enumerate(feature_names):
    val = st.number_input(
        feature,
        value=float(st.session_state.values[i]),
        key=feature
    )
    inputs.append(val)

X_input = np.array(inputs).reshape(1, -1)

# -------- Predict -------- #
# -------- Predict -------- #
if st.button("Predict"):

    pred = model.predict(X_input)[0]

    probs = model.predict_proba(X_input)[0]
    benign_prob = probs[0]
    malignant_prob = probs[1]

    if pred == "M":
        st.error(f"Prediction: MALIGNANT  |  Confidence: {malignant_prob*100:.2f}%")
    else:
        st.success(f"Prediction: BENIGN  |  Confidence: {benign_prob*100:.2f}%")

