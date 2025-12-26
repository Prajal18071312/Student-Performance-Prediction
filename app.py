import streamlit as st
import pandas as pd
import joblib

# Load the trained model and features
model = joblib.load("student_grade_model.pkl")
features = joblib.load("features.pkl")

st.title("Student Performance Prediction")
st.write("Enter the following student details to predict the final grade (G3).")

# Create input fields for features
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}:", min_value=0, step=1)

# Predict button
if st.button("Predict Grade"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Final Grade (G3): {prediction:.2f}")
