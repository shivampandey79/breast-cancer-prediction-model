import streamlit as st
import numpy as np
import pickle

# loading model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.title("Breast Cancer Prediction")

st.write("Enter the required features separated by commas:")

# Taking user input (a single string, split by commas)
input_data = st.text_input("Features", "1.0,2.0,3.0,...")  # Example placeholder

if st.button("Predict"):
    # Convert input into a list of floats
    try:
        features = np.asarray([float(x) for x in input_data.split(',')], dtype=np.float32)
        
        # Check if the input features match the expected number of features for the model
        if features.size == model.n_features_in_:  # Check number of features
            # Prediction
            pred = model.predict(features.reshape(1, -1))
            prediction_message = 'Cancrouse' if pred[0] == 1 else 'Not Cancrouse'
            st.success(f"Prediction: {prediction_message}")
        else:
            st.error(f"Expected {model.n_features_in_} features, but got {features.size}.")
    except ValueError:
        st.error("Please enter valid numeric features separated by commas.")
