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

# Load your car dataset to extract unique manufacturers and models
car_data = pd.read_csv('Cleaned_Car_data.csv')  # Replace with your dataset path

# Extract unique manufacturers and models
unique_manufacturers = car_data['company'].unique().tolist()
unique_models = car_data['name'].unique().tolist()

# Create a Streamlit app
st.title("Car Price Predictor")

# Add UI elements for user input
car_name = st.selectbox("Select Car Name", unique_models)
car_company = st.selectbox("Select Car Company", unique_manufacturers)
car_year = st.number_input("Enter Car Year", min_value=1900, max_value=2050, step=1)
car_kms_driven = st.number_input("Enter KM Driven", min_value=0, max_value=1000000, step=1000)
fuel_type = st.selectbox("Select Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])

if st.button('Predict Car Price'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({'name': [car_name],
                               'company': [car_company],
                               'year': [car_year],
                               'kms_driven': [car_kms_driven],
                               'fuel_type': [fuel_type]})
    
    # Make predictions
    predicted_price = pipe.predict(input_data)
    
    st.write(f'Predicted Car Price: ₹{predicted_price[0]:,.2f}')
