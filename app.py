import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle

model=tf.keras.models.load_model('model.h5')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehotgeo = pickle.load(f)


#streamlit app

st.title("Customer Churn Prediction")

#input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
tenure = st.number_input("Tenure", min_value=0, max_value=10, step=1)
balance = st.number_input("Balance", min_value=0.0, step=0.01)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=0.01)


#prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' :[label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})
   
    
#one-hot encode the geography feature
geo_encoded = onehotgeo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotgeo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

#scale the input data
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")

    st.write("Prediction probability:", prediction_proba)

st.write(" not likely to churn means - low-risk loyal customer")

st.write(" likely to churn means - high-risk customer who may leave ")