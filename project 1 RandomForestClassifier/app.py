import streamlit as st
import pickle
import numpy as np


with open('bitcoin_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Bitcoin Buy/Sell Predictor')
st.write('Enter the details below to predict whether to Buy or Sell Bitcoin today')


volume       = st.number_input('Volume',value=1500000000)
marketcap    = st.number_input('Market Cap',value=900000000000)
close_log    = st.number_input('Close Log',value=10.5)
year         = st.number_input('Year', value=2024)
month        = st.number_input('Month', min_value=1, max_value=12, value=3)
day          = st.number_input('Day',  min_value=1, max_value=31, value=15)
price_range  = st.number_input('Price Range', value=0.08)
momentum     = st.number_input('Momentum',value=0.02)
is_month_end = st.selectbox('Is Month End?', [0, 1])
is_mid_month = st.selectbox('Is Mid Month?',[0, 1])


if st.button('Predict'):
    input_data = np.array([[
        volume, marketcap, close_log, year, month,
        day, price_range, momentum, is_month_end, is_mid_month
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)
    probability  = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success(f'BUY Bitcoin! Confidence: {probability[0][1]:.2%}')
    else:
        st.error(f'SELL / HOLD! Confidence: {probability[0][0]:.2%}')
        