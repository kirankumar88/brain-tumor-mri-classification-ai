import streamlit as st
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import tempfile
import os

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("models/densenet_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------------
# Load Class Names
# -----------------------------
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Sidebar - Model Info
# -----------------------------
st.sidebar.title("Model Information")
st.sidebar.write("**Model:** DenseNet121")
st.sidebar.write("**Test Accuracy:** 91.9%")
st.sidebar.write("**Precision:** 0.92")
st.sidebar.write("**Recall:** 0.92")
st.sidebar.write("**F1 Score:** 0.92")

# Model comparison
st.sidebar.write("### Model Comparison")
data = {
    "Model": ["CNN", "ResNet50", "EfficientNetB0", "DenseNet121"],
    "Accuracy": [0.80, 0.85, 0.89, 0.92]
}
df = pd.DataFrame(data)
st.sidebar.table(df)

# -----------------------------
# Main UI
# -----------------------------
st.title("Brain Tumor MRI Classification")
st.write("Upload MRI image(s) or ZIP file to classify tumor type.")

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
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0])
        ax.set_ylabel("Probability")
        ax.set_xlabel("Class")
        st.pyplot(fig)

        st.markdown("---")

import zipfile
import tempfile
import os
from PIL import Image

st.subheader("Upload ZIP File Containing MRI Images")

uploaded_zip = st.file_uploader("Upload ZIP File", type=["zip"], key="zip_uploader")

if uploaded_zip is not None:
    st.write("ZIP file uploaded. Extracting images...")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, uploaded_zip.name)

        # Save ZIP file
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find images
        image_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, file))

        st.write(f"Found {len(image_files)} images in ZIP.")

        # Predict for each image
        for file_path in image_files:
            image = Image.open(file_path).convert("RGB")
            st.image(image, caption=os.path.basename(file_path), use_container_width=True)

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")

            st.markdown("---")

            st.title("Brain Tumor MRI Classification")

