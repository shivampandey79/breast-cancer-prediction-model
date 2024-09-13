import streamlit as st
import numpy as np
import pickle

# Load the machine learning model using pickle
model = pickle.load(open('model.pkl', 'rb'))

# Create the Streamlit App
st.title("Breast Cancer Prediction")

# Instruction for the user
st.write("Enter the required features separated by commas:")

# Taking user input (a single string, split by commas)
input_data = st.text_input("Features", "1.0,2.0,3.0,...")  # Example placeholder

# Prediction button
if st.button("Predict"):
    try:
        # Convert the input into a list of floats
        features = np.asarray([float(x) for x in input_data.split(',')], dtype=np.float32)

        # Check if the input features match the expected number of features for the model
        if features.size == model.n_features_in_:
            # Make a prediction using the model
            pred = model.predict(features.reshape(1, -1))
            prediction_message = 'Cancrouse' if pred[0] == 1 else 'Not Cancrouse'
            st.success(f"Prediction: {prediction_message}")
        else:
            st.error(f"Expected {model.n_features_in_} features, but got {features.size}.")
    except ValueError:
        st.error("Please enter valid numeric features separated by commas.")
