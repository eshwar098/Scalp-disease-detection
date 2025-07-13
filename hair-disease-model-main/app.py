import os
import requests
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

# Show TensorFlow version
st.text(f"TensorFlow version: {tf.__version__}")

# Download the model if not present
MODEL_URL = 'https://huggingface.co/ravikanth27/hair-disease-model/resolve/main/Hair_Disease.h5'
# MODEL_URL = 'https://huggingface.co/ravikanth27/hair-disease-keras/resolve/main/Hair_Disease.keras'
MODEL_PATH = 'Hair_Disease.h5'

if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model from Hugging Face...'):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        st.success('Model downloaded successfully!')

# Load the model
model = load_model(MODEL_PATH)

# UI Header
st.header('Hair Diseases Classification CNN Model')
Diseases_names = ['Alopecia', 'balness', 'dandruff']

# Prediction function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = '🩺 **Prediction:** The image belongs to **' + Diseases_names[np.argmax(result)] + '** with a confidence of **' + str(round(np.max(result) * 100, 2)) + '%**'
    return outcome

# File upload section
if not os.path.exists('upload'):
    os.makedirs('upload')

uploaded_file = st.file_uploader('📤 Upload an Image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    file_path = os.path.join('upload', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200, caption='🖼️ Uploaded Image')
    st.markdown(classify_images(file_path))
