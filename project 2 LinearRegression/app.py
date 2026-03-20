import streamlit as st
import pickle
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'housing_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'housing_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)



st.title("House Price Prediction System")
st.write("Enter the data bellow:")


area             = st.number_input('Area (sqft)',min_value=500,  max_value=20000, value=7000)
bedrooms         = st.slider('Bedrooms',  min_value=1,    max_value=10,    value=3)
bathrooms        = st.slider('Bathrooms', min_value=1,    max_value=5,     value=2)
stories          = st.slider('Stories (floors)',min_value=1,    max_value=5,     value=2)
parking          = st.slider('Parking spaces', min_value=0,    max_value=5,     value=1)
mainroad         = st.selectbox('Main Road?',  ['Yes', 'No'])
guestroom        = st.selectbox('Guest Room?', ['Yes', 'No'])
basement         = st.selectbox('Basement?',  ['Yes', 'No'])
hotwaterheating  = st.selectbox('Hot Water Heating?', ['Yes', 'No'])
airconditioning  = st.selectbox('Air Conditioning?', ['Yes', 'No'])
prefarea         = st.selectbox('Preferred Area?', ['Yes', 'No'])
furnishingstatus = st.selectbox('Furnishing Status',['Furnished', 'Semi-Furnished', 'Unfurnished'])


def encode(val):
    return 1 if val == 'Yes' else 0

def encode_furnishing(val):
    if val == 'Furnished':      return 2
    elif val == 'Semi-Furnished': return 1
    else:                         return 0


if st.button('Predict Price '):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        encode(mainroad),
        encode(guestroom),
        encode(basement),
        encode(hotwaterheating),
        encode(airconditioning),
        parking,
        encode(prefarea),
        encode_furnishing(furnishingstatus)
    ]])

    input_scaled    = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)
    margin          = predicted_price[0] * 0.15

    st.markdown('---')
    st.subheader('Prediction Result')
    st.success(f' Estimated Price: {predicted_price[0]:,.0f}')
    st.info(f'Price Range: {predicted_price[0]-margin:,.0f} — {predicted_price[0]+margin:,.0f}')
    st.caption('Range based on model error margin of 15%')