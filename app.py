import streamlit as st
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

# Sidebar
st.sidebar.title("Model Information")
st.sidebar.metric("Accuracy", "91.9%")
st.sidebar.metric("Precision", "0.92")
st.sidebar.metric("Recall", "0.92")
st.sidebar.metric("F1 Score", "0.92")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="models/densenet_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Load class names
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def plot_probabilities(prediction):
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0])
    ax.set_ylim([0,1])
    return fig

st.title("Brain Tumor MRI Classification")

# Upload images
uploaded_files = st.file_uploader(
    "Upload MRI Images",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        prediction = predict(image)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        col1, col2 = st.columns([1.5,2])

        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.markdown(f"## Prediction: {predicted_class}")
            st.markdown(f"### Confidence: {confidence*100:.2f}%")
            st.pyplot(plot_probabilities(prediction))

        st.markdown("---")