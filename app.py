import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import base64

def load_model(model_file):
    model = CatBoostClassifier()
    model.load_model(model_file)
    return model

def get_download_link(data, filename):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank">Download CSV File</a>'
    return href

st.title('Credit Fraud Detection')

# Load the pre-trained CatBoost model
model = load_model('trained_catboost_model.cbm')

# Allow users to upload a CSV file
csv_file = st.file_uploader('Upload your CSV file', type=['csv'])

if csv_file:
    # Read the CSV file and display the first few rows
    data = pd.read_csv(csv_file)

    # Process only the first 1000 rows
    data = data[:10000]

    st.write('Data preview:')
    st.write(data.head())

    # Make predictions using the CatBoost model
    with st.spinner('Making predictions...'):
        predictions = model.predict(data)
        data['Class'] = predictions

    # Display the modified data with the 'Class' column
    st.write('Data with predictions:')
    st.write(data.head())

    # Allow users to download the modified CSV file
    st.markdown(get_download_link(data, 'data_with_predictions.csv'), unsafe_allow_html=True)
