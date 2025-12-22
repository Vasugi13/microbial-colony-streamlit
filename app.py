# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
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
        return tf.reshape(
            patches,
            (-1, patches.shape[1] * patches.shape[2], patch_dims)
        )

# ============================================================
# LOAD DL MODELS (OPTIONAL)
# ============================================================
@st.cache_resource
def load_dl_models():
    cnn_model = None
    cnn_vit_model = None

    if os.path.exists("cnn_model.h5"):
        cnn_model = load_model("cnn_model.h5")

    if os.path.exists("cnn_vit_model.h5"):
        cnn_vit_model = load_model(
            "cnn_vit_model.h5",
            custom_objects={"PatchExtract": PatchExtract}
        )

    return cnn_model, cnn_vit_model

# ============================================================
# MAIN PIPELINE
# ============================================================
if uploaded_file is not None:

    # ---------- Load Image ----------
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (512, 512))

    # ---------- Preprocessing ----------
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ---------- Colony Detection ----------
    output = image_np.copy()
    colony_data = []
    colony_info = {"Healthy": 0, "Unhealthy": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)

        health_class = "Healthy" if circularity > 0.55 else "Unhealthy"
        colony_info[health_class] += 1

        color = (0,255,0) if health_class=="Healthy" else (0,0,255)
        cv2.drawContours(output, [cnt], -1, color, 2)

        colony_data.append([area, perimeter, circularity, health_class])

    # ---------- Display ----------
    st.subheader("🔍 Detected Colonies")
    st.image(output, use_column_width=True)

    df = pd.DataFrame(
        colony_data,
        columns=["Area","Perimeter","Circularity","Health_Class"]
    )

    st.subheader("📊 Colony Feature Table")
    st.dataframe(df)

    st.write("📌 Total Colonies:", len(df))
    st.write("🩺 Healthy:", colony_info["Healthy"])
    st.write("⚠️ Unhealthy:", colony_info["Unhealthy"])

    # ============================================================
    # SVM & ANN VALUES (SAFE + NOISE FIT AWARE)
    # ============================================================
    st.subheader("📈 SVM & ANN Values")

    if len(df) < 2:
        st.warning("⚠️ Not enough samples to compute SVM & ANN values")
    else:
        X = df[["Area","Perimeter","Circularity"]]
        y = df["Health_Class"]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        if len(np.unique(y_enc)) < 2:
            st.warning("⚠️ Only one class detected – SVM & ANN values not reliable")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ----- SVM -----
            svm = SVC(kernel="linear", C=0.3)
            svm.fit(X_scaled, y_enc)
            svm_acc = svm.score(X_scaled, y_enc)

            # ----- ANN -----
            ann = MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=1000,
                random_state=42
            )
            ann.fit(X_scaled, y_enc)
            ann_acc = ann.score(X_scaled, y_enc)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("SVM Accuracy", f"{svm_acc:.2f}")
            with col2:
                st.metric("ANN Accuracy", f"{ann_acc:.2f}")

            st.info(
                "ℹ️ These are training accuracies computed on colony features "
                "from the uploaded image. High values may indicate noise fitting."
            )

    # ============================================================
    # CNN & CNN + ViT (OPTIONAL)
    # ============================================================
    cnn_model, cnn_vit_model = load_dl_models()

    st.subheader("🧠 Deep Learning Results")

    if cnn_model:
        img_cnn = cv2.resize(image_np, (128,128)) / 255.0
        img_cnn = np.expand_dims(img_cnn, axis=0)
        cnn_pred = cnn_model.predict(img_cnn)
        cnn_result = "Healthy" if cnn_pred[0][0] > 0.5 else "Unhealthy"
        st.write("🔸 CNN Prediction:", cnn_result)
    else:
        st.info("CNN model not loaded")

    if cnn_vit_model:
        img_vit = cv2.resize(image_np, (224,224)) / 255.0
        img_vit = np.expand_dims(img_vit, axis=0)
        vit_pred = cnn_vit_model.predict(img_vit)
        vit_result = "Healthy" if np.argmax(vit_pred) == 0 else "Unhealthy"
        st.write("🔸 CNN + ViT Prediction:", vit_result)
    else:
        st.info("CNN + ViT model not loaded")

    st.success("✅ Execution completed successfully")
