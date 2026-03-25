import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")

# -----------------------------
# Sidebar Model Info
# -----------------------------
st.sidebar.title("Model Information")
st.sidebar.metric("Accuracy", "91.9%")
st.sidebar.metric("Precision", "0.92")
st.sidebar.metric("Recall", "0.92")
st.sidebar.metric("F1 Score", "0.92")

# -----------------------------
# Paths (Important for Deployment)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "densenet_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.pkl")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# Load Class Names
# -----------------------------
with open(CLASS_PATH, "rb") as f:
    class_names = pickle.load(f)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# -----------------------------
# Probability Plot
# -----------------------------
def plot_probabilities(prediction):
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    return fig

# -----------------------------
# Main App
# -----------------------------
st.title("Brain Tumor MRI Classification")

uploaded_files = st.file_uploader(
    "Upload MRI Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        prediction = predict(image)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        col1, col2 = st.columns([1.5, 2])

        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.markdown(f"## Prediction: {predicted_class}")
            st.markdown(f"### Confidence: {confidence*100:.2f}%")
            st.pyplot(plot_probabilities(prediction))

        st.markdown("---")