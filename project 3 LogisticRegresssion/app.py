import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'telco_churn_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'telco_churn_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

st.title(" Telco Customer Churn Prediction")
st.write("Enter customer details")

tenure= st.slider('Tenure in Months', min_value=0,    max_value=72,   value=24)
monthly_charge  = st.number_input('Monthly Charge ($)', min_value=0.0, max_value=200.0, value=65.0)
contract  = st.selectbox('Contract Type', ['Month-to-Month', 'One Year', 'Two Year'])
satisfaction = st.slider('Satisfaction Score', min_value=1, max_value=5, value=3)
referrals= st.slider('Number of Referrals', min_value=0, max_value=10, value=2)

def encode_contract(val):
    if val == 'Month-to-Month': return 1
    elif val == 'One Year':     return 2
    else:                       return 0

if st.button('Predict Churn'):
    input_data = np.array([[
        tenure,
        monthly_charge,
        encode_contract(contract),
        satisfaction,
        referrals
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)
    probability  = model.predict_proba(input_scaled)[0][1]

    st.markdown('---')
    st.subheader('Prediction Result')

    if prediction[0] == 1:
        st.error(f' This customer is LIKELY TO CHURN — {round(probability*100, 1)}% probability')
    else:
        st.success(f'This customer is LIKELY TO STAY — {round((1-probability)*100, 1)}% probability')

    st.info(f'Churn Probability: {round(probability*100, 1)}%  |  Retention Probability: {round((1-probability)*100, 1)}%')
    st.caption('Logistic Regression model ')
