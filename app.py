import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, Input, LayerNormalization, Add
from tensorflow.keras.utils import to_categorical

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
st.title("🦠 Microbial Colony Count & Health Detection")

uploaded_file = st.file_uploader(
    "Upload Microbial Colony Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ============================================================
    # IMAGE PROCESSING
    # ============================================================
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (512, 512))

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image_np.copy()
    colony_data = []
    colony_info = {"Healthy": 0, "Unhealthy": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)

        health = "Healthy" if circularity > 0.55 else "Unhealthy"
        colony_info[health] += 1

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        colony_data.append([area, perimeter, circularity, health])

    st.subheader("🔍 Detected Colonies")
    st.image(output, use_column_width=True)

    # ============================================================
    # DATAFRAME
    # ============================================================
    df = pd.DataFrame(
        colony_data,
        columns=["Area", "Perimeter", "Circularity", "Health_Class"]
    )
    st.dataframe(df)

    # ============================================================
    # FEATURE PREPARATION
    # ============================================================
    X = df[["Area", "Perimeter", "Circularity"]]
    y = df["Health_Class"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.3, random_state=42
    )

    # ============================================================
    # SVM
    # ============================================================
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    # ============================================================
    # ANN
    # ============================================================
    ann = MLPClassifier(
        hidden_layer_sizes=(25, 10),
        activation="tanh",
        max_iter=1000,
        random_state=24
    )
    ann.fit(X_train, y_train)
    ann_acc = accuracy_score(y_test, ann.predict(X_test))

    # ============================================================
    # CNN
    # ============================================================
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    cnn = Sequential([
        Conv1D(32, 2, activation="relu", input_shape=(3, 1)),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(2, activation="softmax")
    ])

    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cnn.fit(X_train_cnn, y_train_cat, epochs=20, verbose=0)

    cnn_acc = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]

    # ============================================================
    # HYBRID CNN + ATTENTION
    # ============================================================
    def attention_block(x):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        return x

    inp = Input(shape=(3, 1))
    x = Conv1D(32, 2, activation="relu", padding="same")(inp)
    x = attention_block(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(2, activation="softmax")(x)

    hybrid = Model(inp, out)
    hybrid.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hybrid.fit(X_train_cnn, y_train_cat, epochs=20, verbose=0)

    hybrid_acc = hybrid.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]

    # ============================================================
    # RESULTS
    # ============================================================
    st.subheader("📈 Model Accuracy Comparison")

    acc_df = pd.DataFrame({
        "Model": ["SVM", "ANN", "CNN", "Hybrid CNN+Attention"],
        "Accuracy": [svm_acc, ann_acc, cnn_acc, hybrid_acc]
    })

    st.table(acc_df)

    st.success("✅ All Models Executed Successfully")


