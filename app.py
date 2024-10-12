import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set up page title and layout
st.set_page_config(page_title="🔍 Anomaly Detection for Mammograms", layout="wide")
st.title("🔍 Anomaly Detection using Autoencoders for Mammograms")
st.markdown("**Empowering early detection of potential cancerous regions using machine learning.**")

# Sidebar: About section
with st.sidebar:
    st.title("ℹ️ About This App")
    st.write("""
        This app demonstrates how autoencoders can be used for anomaly detection on mammogram images. 
        The autoencoder is trained to reconstruct normal images, and any image that has a high reconstruction loss is flagged as anomalous (potentially cancerous).
    """)
    st.write("Upload your dataset of mammogram images, configure the autoencoder, and run the training process.")
    
    # Sidebar: Upload dataset
    st.title("📁 Upload Dataset")
    uploaded_files = st.file_uploader("Upload mammogram images (JPG/PNG)", accept_multiple_files=True, type=["jpg", "png"])

    # Sidebar: Autoencoder settings
    st.title("⚙️ Autoencoder Settings")
    latent_dim = st.slider("Latent Space Dimension", min_value=16, max_value=256, step=16, value=64)
    epochs = st.slider("Epochs", min_value=5, max_value=100, step=5, value=20)

# Function to load and preprocess images
def preprocess_images(image_files, img_size=(128, 128)):
    images = []
    for img_file in image_files:
        img = Image.open(img_file).convert('L')  # Ensure image is in grayscale
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

# Custom status indicator
def status_indicator(status_text, success=True):
    if success:
        st.success(status_text)
    else:
        st.warning(status_text)

# Main: Train button logic
model_save_path = "FinalModel.keras"
if st.button("🚀 Train Autoencoder") and uploaded_files:
    st.write("### Training Autoencoder...")
    
    # Load and preprocess images
    train_images = preprocess_images(uploaded_files)
    
    if train_images.size == 0:
        status_indicator("No valid images uploaded. Please upload mammogram images.", success=False)
    else:
        st.image(train_images[0], caption="Sample Mammogram Image", use_column_width=True)

        # Check if the model exists
        if os.path.exists(model_save_path):
            autoencoder = load_model(model_save_path)
            status_indicator("Loaded existing autoencoder model.")
        else:
            # Create an autoencoder model
            input_img = layers.Input(shape=train_images.shape[1:])
            x = layers.Flatten()(input_img)
            x = layers.Dense(latent_dim, activation='relu')(x)
            encoded = layers.Dense(latent_dim, activation='relu')(x)

            # Decoder
            x = layers.Dense(np.prod(train_images.shape[1:]), activation='sigmoid')(encoded)
            decoded = layers.Reshape(train_images.shape[1:])(x)

            # Define and compile the autoencoder
            autoencoder = models.Model(input_img, decoded)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')

            # Train the autoencoder
            autoencoder.fit(train_images, train_images, epochs=epochs, verbose=1)

            status_indicator("Autoencoder Training Complete!")
            autoencoder.save(model_save_path)
            st.success(f"Model saved as {model_save_path}")

        # Generate reconstructed images and display them
        reconstructed_images = autoencoder.predict(train_images)
        display_images(train_images[0], reconstructed_images[0])

        # Calculate reconstruction losses
        reconstruction_losses = np.mean(np.square(train_images - reconstructed_images), axis=(1, 2))
        mean_loss = np.mean(reconstruction_losses)
        std_loss = np.std(reconstruction_losses)

        # Define a threshold for anomalies
        threshold = mean_loss + 0.1 * std_loss
        st.write(f"**Mean Loss**: {mean_loss:.6f}, **Standard Deviation**: {std_loss:.6f}, **Threshold**: {threshold:.6f}")

        # Identify anomalous images
        st.write("### 🔍 Anomalous Images Detection")
        anomalous_indices = np.where(reconstruction_losses > threshold)[0]
        
        if anomalous_indices.size > 0:
            for idx in anomalous_indices:
                loss_value = reconstruction_losses[idx].item()  # Get a Python float
                st.write(f"Image {idx} is **anomalous** (Loss: {loss_value:.6f})")
                display_images(train_images[idx], reconstructed_images[idx], title=f"Anomalous Image {idx}")
        else:
            status_indicator("No anomalies detected in the dataset.")
else:
    st.write("Upload images and configure the autoencoder to get started.")

# CSS to improve layout and style
st.markdown("""
    <style>
        .css-1d391kg, .css-1v3fvcr {
            font-size: 16px;
            font-family: 'Roboto', sans-serif;
            background-color: #F7F9FB;
        }
        h1 {
            color: #30475E;
        }
        .stButton>button {
            background-color: #30475E;
            color: white;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #f05454;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)
