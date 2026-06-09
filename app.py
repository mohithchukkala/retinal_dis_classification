import streamlit as st
st.set_page_config(page_title="Retinal Disease Classifier", layout="centered")

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the model (adjust filename as needed)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

# Disease classes (change as per your model's classes)
CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']


st.title("👁️ Retinal Disease Classification")
st.markdown("Upload a retinal fundus image to predict the disease type.")

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.markdown("### 🩺 Prediction Result")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Fundus Image", width=224)

    with col2:
        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    
else:
    st.warning("Please upload an image to continue.")
