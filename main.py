import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# -------------------- LOAD MODELS --------------------
model = tf.keras.models.load_model("model.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("onehot_encoder_geography.pkl", "rb") as f:
    onehotgeo = pickle.load(f)

# -------------------- UI --------------------
st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 25)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0)
num_of_products = st.number_input("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", 0.0)

# -------------------- PREDICTION --------------------
if st.button("Predict"):

    # Encode gender
    gender_encoded = label_encoder.transform([gender])[0]

    # Create dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehotgeo.transform(input_data[["Geography"]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehotgeo.get_feature_names_out(["Geography"])
    )

    # Merge and drop original column
    input_data = pd.concat(
        [input_data.drop("Geography", axis=1), geo_df],
        axis=1
    )

    # Match training column order
    input_data = input_data[scaler.feature_names_in_]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # Output
    if prediction_proba > 0.5:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer is not likely to churn ✅")

    st.write(f"Probability: {prediction_proba:.4f}")