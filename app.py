import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load("model_gst.pkl")

# Retrieve the feature names from the model to ensure correct column order
expected_columns = model.get_booster().feature_names  # Get the expected feature names from the model

# Function to make predictions
def predict(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Rearrange columns to match the model's expected order
    input_df = input_df[expected_columns]

    # Make prediction
    prediction = model.predict(input_df)
    return prediction

# Streamlit app
st.title('Model Prediction App')

# Create input fields for each feature
features = [f'Column{i}' for i in range(22)]  # Column0 to Column21
input_data = {}

# Ensure missing columns are added to match the model's training data
for feature in features:
    input_data[feature] = st.number_input(feature)

# Button to make prediction
if st.button('Predict'):
    prediction = predict(input_data)
    st.write(f'Prediction: {prediction[0]}')
