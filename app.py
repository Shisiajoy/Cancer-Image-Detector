import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json

# Load the autoencoder model (ensure you provide the correct path)
@st.cache(allow_output_mutation=True)
def load_autoencoder():
    autoencoder = tf.keras.models.load_model('FinalModel.keras')
    return autoencoder

# Load threshold from a JSON file (Assuming it was saved during training)
def load_threshold(json_path='threshold.json'):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get("threshold", 0.0075)  # Default threshold if not found
    else:
        return 0.05  # Return a default value if the file is not found

# Preprocess image before feeding to the autoencoder
def preprocess_image(image):
    img = image.resize((224, 224))  # Assuming Autoencoder input size
    img = np.array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Check if the image is grayscale (common for mammogram images)
def is_mammogram(image):
    return len(np.array(image).shape) == 2 or np.array(image).shape[2] == 1  # Grayscale check

# Page title and emoji
st.title("🩺 Mammogram Anomaly Detection 🩺")
st.subheader("detecting anomalies in mammograms! 🎀")

# Sidebar for detailed descriptions
with st.sidebar:
    st.header("Learn More 📖")
    description = st.selectbox(
        "What would you like to know more about?",
        ("Model Overview", "How the AI works", "About Mammogram Images", "Disclaimer")
    )

# Display selected description
if description == "Model Overview":
    st.sidebar.write("""
    **Model Overview:** This autoencoder model is designed to detect anomalies in mammogram images, 
    such as identifying potential signs of breast cancer by comparing the image to healthy samples.
    """)
elif description == "How the AI works":
    st.sidebar.write("""
    **How the AI works:** The model processes the uploaded mammogram image, reconstructs it using the autoencoder,
    and calculates the reconstruction error to determine if it's abnormal.
    """)
elif description == "About Mammogram Images":
    st.sidebar.write("""
    **About Mammogram Images:** Mammograms are X-ray images of the breast used for cancer screening. Our AI helps identify
    any irregularities that might indicate an issue.
    """)
elif description == "Disclaimer":
    st.sidebar.write("""
    **Disclaimer:** This tool is for educational purposes and not for medical diagnosis. Always consult a healthcare professional for accurate diagnosis and treatment.
    """)

# Divider line
st.markdown("---")

# Upload image section
uploaded_file = st.file_uploader("Choose a mammogram image file", type=["jpg", "png", "jpeg"])

# Load the autoencoder model
autoencoder = load_autoencoder()

# Load the threshold from the JSON file (no user input)
threshold = load_threshold()

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Check if it's a mammogram (grayscale or 1 channel)
    if is_mammogram(image):
        st.write("Processing your mammogram image... ⏳")
        try:
            processed_img = preprocess_image(image)
            reconstruction = autoencoder.predict(processed_img)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.abs(processed_img - reconstruction))

            # Use the loaded threshold for anomaly detection
            if reconstruction_error > threshold:
                st.write("🎗️ **Prediction:** This mammogram **may indicate an anomaly**. Please consult a medical professional.")
            else:
                st.write("✅ **Prediction:** This mammogram is likely healthy.")
        except Exception as e:
            st.write(f"⚠️ An error occurred while processing the image: {e}")
    else:
        st.write("⚠️ The uploaded image does not appear to be a valid mammogram (non-grayscale). Please upload a mammogram image.")

# Footer
st.markdown("""
<style>
footer {visibility: hidden;}
footer:after {
    content: 'Made with 💖 by Shisia | Always consult a doctor for medical concerns.';
    visibility: visible;
    display: block;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
