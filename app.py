import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import os
import gdown

# Page configuration
st.set_page_config(page_title="Lung Cancer Prediction App", layout="centered")

# Title & intro
st.title("Lung Cancer Prediction App")
st.markdown("Upload a lung scan image (X-ray/CT) to predict the likelihood of lung cancer.")
st.write("✅ App is running successfully!")

# Google Drive model file
file_id = "1l3vzCgDNfUDRsfzdXoAGceQuNdg1UrWb"
model_url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_path = "LungCancerPrediction.h5"

# Download model if not present
if not os.path.exists(model_path):
    with st.spinner("⏳ Downloading the model..."):
        gdown.download(url=model_url, output=model_path, quiet=False, fuzzy=True)
        st.success("✅ Model downloaded successfully!")

# Load model
with st.spinner("🔍 Loading model..."):
    model = tf.keras.models.load_model(model_path)
st.success("✅ Model loaded!")

# File uploader
uploaded_file = st.file_uploader("Upload a Lung Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("📷 Please upload an image to get a prediction.")
else:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((256, 256))
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Convert RGBA to RGB

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)

    # Predict
    with st.spinner("🤖 Making prediction..."):
        prediction = model.predict(img_array)[0][0]
        result = "🟢 Low Risk of Lung Cancer" if prediction < 0.5 else "🔴 High Risk of Lung Cancer"

    # Show result
    st.subheader("🔬 Prediction Result:")
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence Score:** `{prediction:.2f}`")
