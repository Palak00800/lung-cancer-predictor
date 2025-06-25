import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import os
import gdown

# Google Drive model file URL
model_url = "https://drive.google.com/uc?id=1l3vzCgDNfUDRsfzdXoAGceQuNdg1UrWb"
model_path = "LungCancerPrediction.h5"

# Download model if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# App Page Setup
st.set_page_config(page_title="Lung Cancer Detection App", layout="centered")
st.title("Lung Cancer Prediction App")
st.markdown("Upload a lung scan image (X-ray/CT) to predict the likelihood of lung cancer.")

# Upload Image
uploaded_file = st.file_uploader("Please upload the Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Lung Scan", use_column_width=True)

    # Resize image to match model input shape
    image = image.resize((256, 256))  

    # Convert image to array
    img_array = np.array(image)

    # Convert RGBA → RGB if needed
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # Normalize and add batch dimension
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "Low Risk of Lung Cancer" if prediction < 0.5 else "High Risk of Lung Cancer"

    # Show result
    st.subheader("Prediction Results:")
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence Score:** {prediction:.2f}")
