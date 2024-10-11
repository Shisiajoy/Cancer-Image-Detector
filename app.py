import streamlit as st
import numpy as np
import json
from keras.models import load_model

# Load the autoencoder model
model = load_model('autoencoder_model.keras')

# Load the threshold value
with open('threshold.json', 'r') as json_file:
    threshold_data = json.load(json_file)
    threshold = threshold_data['threshold']

# Streamlit UI code
st.title('Anomaly Detection with Autoencoder')

# Example input for testing the model
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image here (resize, scale, etc.)
    # For example, using OpenCV or PIL
    # Assuming the input image is loaded and processed to be `input_image`

    input_image = ...  # Load and preprocess the uploaded image

    # Predict using the autoencoder
    reconstructed_image = model.predict(np.expand_dims(input_image, axis=0))

    # Calculate the reconstruction error
    reconstruction_error = np.mean((input_image - reconstructed_image[0]) ** 2)

    # Check if the image is an anomaly
    is_anomaly = reconstruction_error > threshold
    st.write(f"Reconstruction Error: {reconstruction_error:.6f}")
    st.write(f"Is Anomaly: {'Yes' if is_anomaly else 'No'}")

    # Display original and reconstructed images
    st.image(input_image, caption='Original Image', use_column_width=True)
    st.image(reconstructed_image[0], caption='Reconstructed Image', use_column_width=True)

