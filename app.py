import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import os

# Page title and layout
st.set_page_config(page_title="Anomaly Detection for Mammograms", layout="centered")
st.title("🔍 Anomaly Detection using Autoencoders for Mammograms")
st.markdown("### Detect anomalies in mammogram images using an autoencoder model trained for reconstruction.")

# Sidebar information and file uploader
st.sidebar.title("Upload Dataset & Configure Model")
st.sidebar.write("Upload your mammogram dataset and set autoencoder parameters to start anomaly detection.")
uploaded_files = st.sidebar.file_uploader("Upload your mammogram images (JPG/PNG)", accept_multiple_files=True, type=["jpg", "png"])

# Sidebar settings
latent_dim = st.sidebar.slider("Latent Space Dimension", min_value=16, max_value=256, step=16, value=64)
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, step=5, value=20)

# Function to load and preprocess images
def preprocess_images(image_files, img_size=(128, 128)):
    images = []
    for img_file in image_files:
        img = Image.open(img_file).convert('L')  # Convert to grayscale
        img = img.resize(img_size)

        # Check if the image is valid and is in the expected size range
        img_array = np.array(img)
        if img_array.size > 0 and img_array.shape[0] > 0 and img_array.shape[1] > 0:
            img_array = img_array / 255.0  # Normalize pixel values
            images.append(img_array)
        else:
            st.warning(f"Image {img_file.name} is not valid. Please upload a valid mammogram image.")
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

# Main section for training the autoencoder
model_save_path = "FinalModel.keras"
if st.sidebar.button("Train Autoencoder") and uploaded_files:
    st.write("### Training Autoencoder...")

    # Load and preprocess images
    train_images = preprocess_images(uploaded_files)

    if train_images.size == 0:
        st.warning("No valid images uploaded. Please upload mammogram images.")
    else:
        st.image(train_images[0], caption="Sample Mammogram Image", use_column_width=True)

        # Check if the model exists
        if os.path.exists(model_save_path):
            autoencoder = load_model(model_save_path)
            st.write("Loaded existing autoencoder model.")
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
            st.success("Autoencoder Training Complete!")
            autoencoder.save(model_save_path)

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
        st.write("### Anomalous Images Detection")
        anomalous_indices = np.where(reconstruction_losses > threshold)[0]
        
        if anomalous_indices.size > 0:
            for idx in anomalous_indices:
                loss_value = reconstruction_losses[idx].item()  # Get a Python float
                st.write(f"Image {idx} is anomalous (Loss: {loss_value:.6f})")
                display_images(train_images[idx], reconstructed_images[idx], title=f"Anomalous Image {idx}")
        else:
            st.success("No anomalies detected in the dataset.")

else:
    st.write("Upload images and configure the autoencoder to get started.")
