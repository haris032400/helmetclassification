# streamlit_helmet_app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🏍️Helmet Detection")
st.write("Upload an image to check if a person is wearing a helmet.")

# Load model
model = load_model('helmet_model.h5')  # Update path if needed
IMG_SIZE = 128
CLASS_NAMES = ['Dont Wear Helmet', 'Wear Helmet']

def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return CLASS_NAMES[class_index]

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        try:
            result = predict_image(image)
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error: {e}")


