import streamlit as st
from PIL import Image

# Set up the page title and layout
st.set_page_config(page_title="Simple Streamlit Test", layout="centered")
st.title("🧪 Simple Streamlit Test App")

# Sidebar with details
st.sidebar.title("About")
st.sidebar.write("""
This is a basic Streamlit app to test if everything is working fine.
You can upload an image, and it will be displayed below.
""")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])

# If an image is uploaded, display it
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display some info about the image
    st.write("### Image details:")
    st.write(f"File Name: {uploaded_file.name}")
    st.write(f"Image Format: {image.format}")
    st.write(f"Image Size: {image.size}")
else:
    st.write("Upload an image to see it displayed here.")

# Custom CSS
st.markdown("""
    <style>
    .css-1d391kg, .css-1v3fvcr {
        font-size: 16px;
        font-family: 'Roboto', sans-serif;
        background-color: #F0F2F6;
    }
    </style>
""", unsafe_allow_html=True)
