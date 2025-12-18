import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, Input, LayerNormalization, Add
from tensorflow.keras.utils import to_categorical
from PIL import Image

st.set_page_config(page_title="Microbial Colony Detection", layout="wide")
st.title("🧫 Microbial Colony Detection & Classification")

uploaded_file = st.file_uploader(
    "Upload Petri Dish Image", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    colony_info = {"Small": 0, "Medium": 0, "Large": 0, "Healthy": 0, "Unhealthy": 0}
    colony_data = []

    for i, cnt in enumerate(contours):
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

        health_class = "Healthy" if circularity > 0.55 else "Unhealthy"

        colony_info[size_class] += 1
        colony_info[health_class] += 1

        cv2.drawContours(output, [cnt], -1, color, 2)

        colony_data.append({
            "Area": area,
            "Perimeter": perimeter,
            "Circularity": circularity,
            "Health_Class": health_class
        })

    df = pd.DataFrame(colony_data)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Colonies")
        st.image(output, use_column_width=True)

    with col2:
        st.subheader("Colony Summary")
        st.write(colony_info)

    # ---------------- ML MODELS ----------------
    features = df[["Area", "Perimeter", "Circularity"]]
    labels = df["Health_Class"]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = StandardScaler().fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    ann = MLPClassifier(hidden_layer_sizes=(25, 10), max_iter=1000)
    ann.fit(X_train, y_train)
    ann_acc = accuracy_score(y_test, ann.predict(X_test))

    X_train_cnn = X_train.reshape(-1, 3, 1)
    X_test_cnn = X_test.reshape(-1, 3, 1)
    y_train_cnn = to_categorical(y_train)
    y_test_cnn = to_categorical(y_test)

    cnn = Sequential([
        Conv1D(32, 2, activation='relu', input_shape=(3,1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train_cnn, y_train_cnn, epochs=20, verbose=0)
    cnn_acc = cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)[1]

    st.subheader("📊 Model Accuracies")
    st.write({
        "SVM": round(svm_acc, 2),
        "ANN": round(ann_acc, 2),
        "CNN": round(cnn_acc, 2)
    })

    fig, ax = plt.subplots()
    ax.bar(["SVM", "ANN", "CNN"], [svm_acc, ann_acc, cnn_acc])
    ax.set_ylim(0, 1)
    st.pyplot(fig)
