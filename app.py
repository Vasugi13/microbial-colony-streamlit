import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ============================================================
# FIX RANDOMNESS (IMPORTANT FOR CONSISTENT OUTPUT)
# ============================================================
np.random.seed(42)
random.seed(42)

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
    # IMAGE LOADING
    # ============================================================
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (512, 512))

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        morphed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = image_np.copy()

    colony_info = {
        "Small": 0,
        "Medium": 0,
        "Large": 0,
        "Healthy": 0,
        "Unhealthy": 0
    }

    colony_data = []

    # ============================================================
    # COLONY DETECTION & CLASSIFICATION
    # ============================================================
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)

        if area < 180:
            size_class = "Small"
            color = (0, 255, 0)
        elif area < 600:
            size_class = "Medium"
            color = (255, 255, 0)
        else:
            size_class = "Large"
            color = (255, 0, 0)

        # Fixed rule for consistent output
        health_class = "Healthy" if circularity > 0.55 else "Unhealthy"

        colony_info[size_class] += 1
        colony_info[health_class] += 1

        cv2.drawContours(output, [cnt], -1, color, 2)

        colony_data.append({
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2),
            "Circularity": round(circularity, 3),
            "Health_Class": health_class
        })

    # ============================================================
    # DISPLAY DETECTED IMAGE
    # ============================================================
    st.subheader("🔍 Detected Colonies")
    st.image(output, use_column_width=True)

    # ============================================================
    # DATAFRAME
    # ============================================================
    df = pd.DataFrame(colony_data)
    st.subheader("📊 Colony Feature Table")
    st.dataframe(df)

    # ============================================================
    # PIE CHARTS
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [colony_info["Small"], colony_info["Medium"], colony_info["Large"]],
            labels=["Small", "Medium", "Large"],
            autopct="%1.1f%%"
        )
        ax1.set_title("Size Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(
            [colony_info["Healthy"], colony_info["Unhealthy"]],
            labels=["Healthy", "Unhealthy"],
            autopct="%1.1f%%"
        )
        ax2.set_title("Health Distribution")
        st.pyplot(fig2)

    # ============================================================
    # MACHINE LEARNING
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

    # ---------------- SVM (BASELINE) ----------------
    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    # ---------------- ANN (IMPROVED – HIGHER ACCURACY) ----------------
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate="adaptive",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    ann.fit(X_train, y_train)
    ann_acc = accuracy_score(y_test, ann.predict(X_test))

    # ============================================================
    # ACCURACY DISPLAY
    # ============================================================
    st.subheader("📈 Model Accuracy Comparison")
    st.write(f"**SVM Accuracy:** {svm_acc:.2f}")
    st.write(f"**ANN Accuracy:** {ann_acc:.2f} ✅")

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(["SVM", "ANN"], [svm_acc, ann_acc])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_title("Model Performance Comparison")
    st.pyplot(fig_bar)

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================
    st.subheader("🧩 Confusion Matrix (ANN)")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, ann.predict(X_test), ax=ax_cm
    )
    st.pyplot(fig_cm)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    st.success("✅ Analysis Completed Successfully")
    st.write("**Total Colonies Detected:**", len(df))
    st.write("**Healthy Colonies:**", colony_info["Healthy"])
    st.write("**Unhealthy Colonies:**", colony_info["Unhealthy"])



