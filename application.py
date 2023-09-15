import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Load your saved machine learning model
pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Create a Streamlit app
st.title("Price Predictor")

# Add UI elements for user input
car_name = st.text_input("Car Name")
car_company = st.text_input("Car Company")
car_year = st.number_input("Launch Year", min_value=1980, max_value=2025, step=1)
car_kms_driven = st.number_input("KM Driven", min_value=0, max_value=1000000, step=100)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])

if st.button('Predict Car Price'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({'name': [car_name],
                               'company': [car_company],
                               'year': [car_year],
                               'kms_driven': [car_kms_driven],
                               'fuel_type': [fuel_type]})
    
    # Make predictions
    predicted_price = pipe.predict(input_data)
    
    st.write(f'Predicted Car Price: {predicted_price[0]:,.2f}')
