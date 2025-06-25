import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import os
import gdown

# Use full export URL
file_id = "1l3vzCgDNfUDRsfzdXoAGceQuNdg1UrWb"
model_url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_path = "LungCancerPrediction.h5"

# Download the model if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(url=model_url, output=model_path, quiet=False, fuzzy=True)

# Load the model
model = tf.keras.models.load_model(model_path)
