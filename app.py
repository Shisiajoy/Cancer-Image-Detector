import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set up the page title and layout
st.set_page_config(page_title="Anomaly Detection with Autoencoders", layout="centered")
st.title("🔍 Anomaly Detection using Autoencoders for Mammograms")

# Sidebar with details
st.sidebar.title("About")
st.sidebar.write("""
This app demonstrates anomaly detection using an autoencoder for mammograms.
Upload your mammogram images, and we'll detect anomalies based on reconstruction loss.
""")

# Sidebar contact
st.sidebar.title("Contact")
st.sidebar.info("""
Developed by [Shisia Joy](https://github.com/shisia), feel free to reach out for any inquiries!
""")

# Function to load and preprocess images
def preprocess_images(image_files, img_size=(128, 128)):  # Updated image size to 128x128 for better quality
    images = []
    for img_file in image_files:
        img = Image.open(img_file)

        # Check if image is grayscale (as mammograms are usually grayscale)
        if img.mode != 'L':
            st.error(f"Error: {img_file.name} is not a grayscale image. Please upload mammogram images only.")
            return None

        img = img.resize(img_size)
        img_array = np.array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
    return np.array(images)

# Function to display images
def display_images(original, reconstructed, title="Reconstruction"):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    st.write(f"### {title}")
    st.pyplot(fig)

# Load the pre-trained model
model_save_path = "FinalModel.keras"
if os.path.exists(model_save_path):
    autoencoder = load_model(model_save_path)
    st.sidebar.success(f"Loaded pre-trained model: {model_save_path}")
else:
    st.sidebar.error(f"Error: Pre-trained model {model_save_path} not found.")

# Set the threshold from a saved file or define it manually
threshold_save_path = "threshold.txt"
if os.path.exists(threshold_save_path):
    with open(threshold_save_path, "r") as f:
        threshold = float(f.read().strip())
    st.sidebar.success(f"Loaded threshold: {threshold}")
else:
    threshold = 0.001  # Set a default threshold if not available
    st.sidebar.warning(f"Using default threshold: {threshold}")

# Sidebar to upload dataset
st.sidebar.title("Upload Dataset")
uploaded_files = st.sidebar.file_uploader("Upload your mammogram images (JPG/PNG)", accept_multiple_files=True, type=["jpg", "png"])

# Run inference with the pre-trained model
if st.sidebar.button("Run Anomaly Detection") and uploaded_files:

    st.write("### Running Anomaly Detection...")
    # Load and preprocess images
    test_images = preprocess_images(uploaded_files)

    # Check if preprocessing returned valid images
    if test_images is None:
        st.warning("Please upload valid mammogram images to proceed.")
    else:
        # Display the first image
        st.image(test_images[0], caption="Sample Mammogram Image from Dataset", use_column_width=True)

        # Generate reconstructed images and display them
        reconstructed_images = autoencoder.predict(test_images)
        display_images(test_images[0], reconstructed_images[0])

        # Calculate reconstruction losses
        reconstruction_losses = np.mean(np.square(test_images - reconstructed_images), axis=(1, 2))
        mean_loss = np.mean(reconstruction_losses)
        std_loss = np.std(reconstruction_losses)

        st.write(f"Mean Loss: {mean_loss:.6f}, Standard Deviation: {std_loss:.6f}, Threshold: {threshold:.6f}")

        # Identify anomalous images
        st.write("### Anomalous Images Detection")
        anomalous_indices = np.where(reconstruction_losses > threshold)[0]
        if len(anomalous_indices) > 0:
            for idx in anomalous_indices:
                st.write(f"Image {idx} is anomalous (Loss: {reconstruction_losses[idx]:.6f})")
                display_images(test_images[idx], reconstructed_images[idx], title=f"Anomalous Image {idx}")
        else:
            st.write("No anomalies detected in the dataset.")

else:
    st.write("Upload images and run the anomaly detection to get started.")

# Add some custom CSS to style the app
st.markdown("""
    <style>
    .css-1d391kg, .css-1v3fvcr {
        font-size: 16px;
        font-family: 'Roboto', sans-serif;
        background-color: #F0F2F6;
    }
    </style>
""", unsafe_allow_html=True)
