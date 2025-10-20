import os
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Set page title
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")

# Upload folder (Streamlit uploads are temporary)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
@st.cache_resource
def load_model():
    base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(base_model.inputs, output)
    model.load_weights('vgg_unfrozen.h5')
    return model

model_03 = load_model()
st.success("Model loaded successfully!")

# Helper functions
def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"

def getResult(img_path):
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((128, 128))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0) / 255.0  # normalize
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)[0]
    return result01

# Streamlit UI
st.title("Pneumonia Detection Using Deep Learning 🫁")
st.write("Upload a chest X-ray image to check if it shows Pneumonia or is Normal.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Analyzing..."):
        prediction = getResult(file_path)
        result = get_className(prediction)
        st.success(f"Prediction: {result}")
