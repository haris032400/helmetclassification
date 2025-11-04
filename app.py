# helmet_app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# --- Constants ---
IMG_SIZE = 128
CLASS_NAMES = ['Dont Wear Helmet', 'Wear Helmet']  # Update if needed
MODEL_PATH = "helmet_model.h5"  # Ensure this path is correct
LOGO_PATH = "logo.png"  # Place your logo in the same folder

# --- Load Model ---
@st.cache_resource  # Cache the model to speed up reloads
def load_helmet_model():
    return load_model(MODEL_PATH)

model = load_helmet_model()

# --- Page Layout ---
st.set_page_config(page_title="Helmet Detection", page_icon="🪖")

# Display logo + title
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_PATH, width=80)
with col2:
    st.markdown("## 🪖 Helmet Detection")
st.write("Upload an image to check if a person is wearing a helmet.")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_image(image):
    img = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    return CLASS_NAMES[class_index]

# --- Prediction ---
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        try:
            result = predict_image(uploaded_file)
            st.success(f"Predicted: **{result}**")
        except Exception as e:
            st.error(f"Error in prediction: {e}")



