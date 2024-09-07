import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Helper function to encode the image in base64
import base64
def get_base64_of_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom CSS for background image, white text, black input boxes, and blue buttons
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{get_base64_of_image('_d2c01da9-872c-4471-b213-219d0475a3a4.jpeg')}");
            background-size: cover;
            color: white;  /* Default text color */
        }}

        /* Style for title */
        .stApp h1 {{
            color: white !important;  /* White title text */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);  /* Add shadow to title */
            font-size: 40px;
            font-weight: bold;
        }}

        /* Make input label text white */
        .stNumberInput label {{
            color: white !important;  /* Set label color (Feature V1, Feature V2, etc.) to white */
        }}

        /* Set input field background to black and text to white */
        .stNumberInput input {{
            background-color: black !important;  /* Set the entire input field background to black */
            color: white !important;  /* Set the input text color to white */
            border: 1px solid white !important;  /* Optional: Add a white border */
            border-radius: 8px !important;  /* Add rounded corners to input boxes */
        }}

        /* Style the buttons */
        .stButton>button {{
            background-color: #00b4d8;  /* Blue button background */
            color: white;  /* Button text color */
            border-radius: 8px;  /* Rounded buttons */
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        }}

        .stButton>button:hover {{
            background-color: #0077b6;  /* Darker blue button on hover */
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the background setter function
set_bg()

# Load the model
model = load_model('fraud_detection_model.h5')

# Streamlit App
st.title('Credit Card Fraud Detection')

# File uploader for the CSV dataset
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Display the dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head())  # Show the first few rows
    
    # Select features for prediction
    st.write("### Enter Transaction Features")
    features = []
    for i in range(1, 29):
        feature = st.number_input(f"Feature V{i}", step=0.01)
        features.append(feature)
    
    amount = st.number_input("Transaction Amount", step=0.01)
    time = st.number_input("Transaction Time", step=0.01)
    
    # Add the Amount and Time to the feature list
    features.append(amount)
    features.append(time)

    # Reshape the input data to match the model's expected shape (1, 30, 1)
    new_transaction = np.array(features).reshape(1, 30, 1)
    
    # Make a prediction
    if st.button('Predict'):
        prediction = model.predict(new_transaction)
        output = 1 if prediction >= 0.5 else 0  # Fraud or non-fraud
        if output == 1:
            st.error('⚠️ Fraud Detected!')
        else:
            st.success('✅ No Fraud Detected.')
else:
    st.info('Please upload a CSV file to proceed.')
