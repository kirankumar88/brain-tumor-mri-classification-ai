import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
}
.pred-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar - Model Info
# -----------------------------
st.sidebar.title("Model Information")
st.sidebar.markdown("### DenseNet121")
st.sidebar.markdown("**Accuracy:** 91.9%")
st.sidebar.markdown("**Precision:** 0.92")
st.sidebar.markdown("**Recall:** 0.92")
st.sidebar.markdown("**F1 Score:** 0.92")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_my_model():
    return load_model("models/densenet_model.keras")

model = load_my_model()

# Load class names
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Title
# -----------------------------
st.title("Brain Tumor MRI Classification")

# -----------------------------
# Multiple Image Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload MRI Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        col1, col2 = st.columns([1.5, 2])

        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.markdown(f"## Prediction: {predicted_class}")
            st.markdown(f"### Confidence: {confidence*100:.2f}%")

            fig, ax = plt.subplots()
            ax.bar(class_names, prediction[0])
            ax.set_ylabel("Probability")
            ax.set_xlabel("Class")
            st.pyplot(fig)

        st.markdown("---")

# -----------------------------
# ZIP Upload
# -----------------------------
st.subheader("Upload ZIP File Containing MRI Images")

uploaded_zip = st.file_uploader("Upload ZIP File", type=["zip"], key="zip_uploader")

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, uploaded_zip.name)

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        image_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, file))

        st.write(f"Found {len(image_files)} images in ZIP.")

        for file_path in image_files:
            image = Image.open(file_path).convert("RGB")

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            col1, col2 = st.columns([1.5, 2])

            with col1:
                st.image(image, caption=os.path.basename(file_path), use_container_width=True)

            with col2:
                st.markdown(f"## Prediction: {predicted_class}")
                st.markdown(f"### Confidence: {confidence*100:.2f}%")

                fig, ax = plt.subplots()
                ax.bar(class_names, prediction[0])
                ax.set_ylabel("Probability")
                ax.set_xlabel("Class")
                st.pyplot(fig)

            st.markdown("---")