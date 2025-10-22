import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn based on their details.")

# ℹ️ Info section
with st.expander("ℹ️ How this model works"):
    st.markdown("""
    This model predicts the **likelihood of a customer leaving (churning)** 
    based on their account and usage details.

    Here's a quick breakdown:
    - The model was trained on **telecom customer data** using **Logistic Regression**.
    - Input features such as `Contract Type`, `Internet Service`, and `Monthly Charges`
      are preprocessed using **One-Hot Encoding** and **Standard Scaling**.
    - The model outputs a **probability** that a customer will churn.
    - If the probability > 0.5 → The app predicts **"Customer will churn"**.
      Otherwise → **"Customer will stay"**.

    🧠 **ML Algorithm:** Logistic Regression  
    🧰 **Libraries Used:** scikit-learn, pandas, streamlit  
    """)

# Load model artifacts
model = pickle.load(open(r"model/churn_model.pkl", "rb"))
scaler = pickle.load(open(r"model/scaler.pkl", "rb"))
feature_names = pickle.load(open(r"model/feature_names.pkl", "rb"))

# User inputs
st.header("Enter Customer Details:")
tenure = st.number_input("Tenure (in months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Create a blank DataFrame with all features = 0
user_df = pd.DataFrame(columns=feature_names)
user_df.loc[0] = 0

# Fill user input into relevant features
if 'tenure' in user_df.columns:
    user_df.at[0, 'tenure'] = tenure
if 'MonthlyCharges' in user_df.columns:
    user_df.at[0, 'MonthlyCharges'] = monthly_charges

# Handle contract type (training dropped first dummy, so base case = Month-to-month)
if contract == "One year" and 'Contract_One year' in user_df.columns:
    user_df.at[0, 'Contract_One year'] = 1
elif contract == "Two year" and 'Contract_Two year' in user_df.columns:
    user_df.at[0, 'Contract_Two year'] = 1
# If contract == "Month-to-month", leave all zeros (baseline category)

# Scale and predict
scaled_input = scaler.transform(user_df)
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    st.success("🚫 Customer is likely to churn" if prediction == 1 else "✅ Customer is likely to stay")
