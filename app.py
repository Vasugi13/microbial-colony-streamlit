# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
import urllib.request
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import tensorflow as tf

# ============================================================
# FIX RANDOMNESS
# ============================================================
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="Microbial Colony Detection", layout="wide")
st.title("🦠 Microbial Colony Detection & Health Analysis")

uploaded_file = st.file_uploader(
    "Upload Microbial Colony Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PATCH EXTRACT LAYER FOR ViT
# ============================================================
class PatchExtract(layers.Layer):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, (-1, patches.shape[1]*patches.shape[2], patch_dims))

# ============================================================
# DOWNLOAD MODELS FROM URL IF NOT PRESENT
# ============================================================
def download_model(url, filename):
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        st.success(f"{filename} downloaded successfully!")

# Replace these URLs with your actual model URLs
CNN_MODEL_URL = ""
CNN_VIT_MODEL_URL = ""

download_model(CNN_MODEL_URL, "cnn_model.h5")
download_model(CNN_VIT_MODEL_URL, "cnn_vit_model.h5")

# ============================================================
# LOAD DEEP MODELS
# ============================================================
@st.cache_resource
def load_dl_models():
    cnn_model = load_model("cnn_model.h5")
    cnn_vit_model = load_model("cnn_vit_model.h5", custom_objects={"PatchExtract": PatchExtract})
    return cnn_model, cnn_vit_model

# ============================================================
# PROCESS IMAGE AND RUN MODELS
# ============================================================
if uploaded_file is not None:

    # -----------------------------
    # IMAGE PROCESSING
    # -----------------------------
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (512, 512))

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image_np.copy()

    colony_info = {"Small":0,"Medium":0,"Large":0,"Healthy":0,"Unhealthy":0}
    colony_data = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)
        if area < 180:
            size_class = "Small"
            color = (0,255,0)
        elif area < 600:
            size_class = "Medium"
            color = (255,255,0)
        else:
            size_class = "Large"
            color = (255,0,0)
        health_class = "Healthy" if circularity > 0.55 else "Unhealthy"
        colony_info[size_class] += 1
        colony_info[health_class] += 1
        cv2.drawContours(output,[cnt],-1,color,2)
        colony_data.append([area, perimeter, circularity, health_class])

    # -----------------------------
    # DISPLAY IMAGE
    # -----------------------------
    st.subheader("🔍 Detected Colonies")
    st.image(output, use_column_width=True)

    # -----------------------------
    # DATAFRAME
    # -----------------------------
    df = pd.DataFrame(colony_data, columns=["Area","Perimeter","Circularity","Health_Class"])
    st.subheader("📊 Colony Feature Table")
    st.dataframe(df)

    # -----------------------------
    # SVM & ANN
    # -----------------------------
    X = df[["Area","Perimeter","Circularity"]]
    y = df["Health_Class"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel="linear", C=0.3)
    svm.fit(X_scaled, y_enc)
    svm_acc = svm.score(X_scaled, y_enc)

    ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                        max_iter=1500, random_state=42)
    ann.fit(X_scaled, y_enc)
    ann_acc = ann.score(X_scaled, y_enc)

    # -----------------------------
    # CNN & CNN+ViT Prediction
    # -----------------------------
    cnn_model, cnn_vit_model = load_dl_models()

    img_cnn = cv2.resize(image_np, (128,128))/255.0
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_pred = cnn_model.predict(img_cnn)
    cnn_result = "Healthy" if cnn_pred[0][0] > 0.5 else "Unhealthy"

    img_vit = cv2.resize(image_np, (224,224))/255.0
    img_vit = np.expand_dims(img_vit, axis=0)
    vit_pred = cnn_vit_model.predict(img_vit)
    vit_result = "Healthy" if vit_pred[0][0] > 0.5 else "Unhealthy"

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("📈 Model Results")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹 **SVM Accuracy:**", f"{svm_acc:.2f}")
        st.write("🔹 **ANN Accuracy:**", f"{ann_acc:.2f}")
    with col2:
        st.write("🧠 **CNN Prediction:**", cnn_result)
        st.write("🤖 **CNN + ViT Prediction:**", vit_result)

    st.success("✅ All Models Executed Successfully")
    st.write("📌 **Total Colonies Detected:**", len(df))
    st.write("🩺 **Healthy Colonies:**", colony_info["Healthy"])
    st.write("⚠️ **Unhealthy Colonies:**", colony_info["Unhealthy"])


