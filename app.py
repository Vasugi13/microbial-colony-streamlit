import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Microbial Colony Detection", layout="wide")
st.title("🦠 Microbial Colony Count & Health Detection")

uploaded_file = st.file_uploader("Upload Microbial Colony Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # ============================================================
    # IMAGE LOADING
    # ============================================================
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image_np.copy()
    colony_info = {"Small": 0, "Medium": 0, "Large": 0, "Healthy": 0, "Unhealthy": 0}
    colony_data = []

    # ============================================================
    # COLONY DETECTION
    # ============================================================
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)

        if area < 180:
            size = "Small"
            color = (0, 255, 0)
        elif area < 600:
            size = "Medium"
            color = (255, 255, 0)
        else:
            size = "Large"
            color = (255, 0, 0)

        health = "Healthy" if circularity > 0.55 else "Unhealthy"

        colony_info[size] += 1
        colony_info[health] += 1

        cv2.drawContours(output, [cnt], -1, color, 2)

        colony_data.append({
            "Area": area,
            "Perimeter": perimeter,
            "Circularity": circularity,
            "Size_Class": size,
            "Health_Class": health
        })

    st.subheader("🔍 Detected Colonies")
    st.image(output, use_column_width=True)

    # ============================================================
    # DATAFRAME
    # ============================================================
    df = pd.DataFrame(colony_data)
    st.subheader("📊 Colony Data")
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
    features = df[["Area", "Perimeter", "Circularity"]]
    labels = df["Health_Class"]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # SVM
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    # ANN
    ann = MLPClassifier(hidden_layer_sizes=(25, 10), max_iter=1000)
    ann.fit(X_train, y_train)
    ann_acc = accuracy_score(y_test, ann.predict(X_test))

    # ============================================================
    # RESULTS
    # ============================================================
    st.subheader("📈 Model Accuracy")
    st.write(f"**SVM Accuracy:** {svm_acc:.2f}")
    st.write(f"**ANN Accuracy:** {ann_acc:.2f}")

    st.subheader("🧩 Confusion Matrix (SVM)")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, svm.predict(X_test), ax=ax_cm)
    st.pyplot(fig_cm)

    # ============================================================
    # SUMMARY
    # ============================================================
    st.success("✅ Analysis Completed Successfully")
    st.write("**Total Colonies:**", len(df))
    st.write("**Healthy Colonies:**", colony_info["Healthy"])
    st.write("**Unhealthy Colonies:**", colony_info["Unhealthy"])

