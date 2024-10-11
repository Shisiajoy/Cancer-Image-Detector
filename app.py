import streamlit as st
import numpy as np
import json
from keras.models import load_model
from PIL import Image
import cv2

# Load the autoencoder model
model = load_model('autoencoder_model.keras')

# Load the threshold value
with open('threshold.json', 'r') as json_file:
    threshold_data = json.load(json_file)
    threshold = threshold_data['threshold']

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert to grayscale (if it's not already)
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize the image to 128x128
    image = image.resize((128, 128))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 128, 128, 1)
    
    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Shape: (1, 128, 128, 1)
    
    return image_array

# Streamlit UI code
st.title('Anomaly Detection with Autoencoder')

# Upload an image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    input_image = preprocess_image(image)

    # Predict using the autoencoder
    reconstructed_image = model.predict(input_image)

    # Calculate the reconstruction error
    reconstruction_error = np.mean((input_image - reconstructed_image) ** 2)

    # Check if the image is an anomaly
    is_anomaly = reconstruction_error > threshold
    st.write(f"Reconstruction Error: {reconstruction_error:.6f}")
    st.write(f"Is Anomaly: {'Yes' if is_anomaly else 'No'}")

    # Display original and reconstructed images
    st.image(image, caption='Original Image', use_column_width=True)
    st.image(reconstructed_image[0].squeeze(), caption='Reconstructed Image', use_column_width=True, clamp=True)
