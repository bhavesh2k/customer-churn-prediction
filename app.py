import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn based on their details.")

# Load model and scaler
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# User inputs
st.header("Enter Customer Details:")
tenure = st.number_input("Tenure (in months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Convert input to DataFrame
user_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "Contract_Month-to-month": [1 if contract == "Month-to-month" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
})

# Scale input
scaled_input = scaler.transform(user_data)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    st.success("🚫 Customer is likely to churn" if prediction == 1 else "✅ Customer is likely to stay")
