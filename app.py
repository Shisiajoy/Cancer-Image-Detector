import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Load the trained model (ensure you provide the path to your model)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('FinalModel.keras')
    return model

# Preprocess image before feeding to the model
def preprocess_image(image):
    img = image.resize((224, 224))  # Assuming MobileNetV2 input size
    img = np.array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Cute page title and emoji
st.title("🩺 Mammogram Cancer Detection 🩺")
st.subheader("Your trusty AI companion for breast cancer screening!")
st.write("Upload a mammogram image below to check for signs of cancer! 🎀")

# Upload image section
uploaded_file = st.file_uploader("Choose a mammogram image file", type=["jpg", "png", "jpeg"])

# Drop-down for detailed descriptions
description = st.selectbox(
    "What would you like to know more about?",
    ("Model Overview", "How the AI works", "About Mammogram Images", "Disclaimer")
)

# Display selected description
if description == "Model Overview":
    st.write("""
    **Model Overview:** This is a convolutional neural network (CNN) built using MobileNetV2 architecture, 
    fine-tuned for mammogram image analysis to detect signs of breast cancer.
    """)
elif description == "How the AI works":
    st.write("""
    **How the AI works:** The model processes the uploaded mammogram image, scales it down, and passes it 
    through multiple layers of convolution and activation to identify patterns that may indicate cancer.
    """)
elif description == "About Mammogram Images":
    st.write("""
    **About Mammogram Images:** Mammograms are X-ray images of the breast used for cancer screening. They help 
    detect abnormalities, and our AI is trained to identify signs of malignancy.
    """)
elif description == "Disclaimer":
    st.write("""
    **Disclaimer:** This tool is intended for educational purposes and not for medical diagnosis. Always consult 
    a healthcare professional for accurate diagnosis and treatment.
    """)

# Cute divider line
st.markdown("---")

# Load the model
model = load_model()

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Check if it's a mammogram
    if "mammogram" in uploaded_file.name.lower():
        st.write("Processing your mammogram image... ⏳")
        try:
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            # Example threshold for cancer detection
            if prediction[0][0] > 0.5:
                st.write("🎗️ **Prediction:** This mammogram **may indicate cancer**. Please consult a medical professional.")
            else:
                st.write("✅ **Prediction:** This mammogram does not indicate cancer. Regular screening is encouraged.")
        except Exception as e:
            st.write(f"⚠️ An error occurred while processing the image: {e}")
    else:
        st.write("⚠️ The uploaded file is not a mammogram image. Please upload a valid mammogram image.")

# Cute footer
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
