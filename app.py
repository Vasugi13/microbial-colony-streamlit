import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

st.set_page_config("Microbial Colony Detection", layout="wide")
st.title("🧫 Microbial Colony Detection & Classification")

uploaded_file = st.file_uploader("Upload Petri Dish Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours,_ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colony_info = {"Small":0,"Medium":0,"Large":0,"Healthy":0,"Unhealthy":0}
    data = []
    output = image.copy()

    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        peri = cv2.arcLength(cnt,True)
        circ = 4*np.pi*area/(peri**2+1e-5)

        if area < 180:
            size = "Small"
            color = (0,255,0)
        elif area < 600:
            size = "Medium"
            color = (255,255,0)
        else:
            size = "Large"
            color = (255,0,0)

        health = "Healthy" if circ > 0.55 else "Unhealthy"

        colony_info[size]+=1
        colony_info[health]+=1

        cv2.drawContours(output,[cnt],-1,color,2)

        data.append({
            "Area":area,
            "Perimeter":peri,
            "Circularity":circ,
            "Health":health
        })

    df = pd.DataFrame(data)

    col1,col2 = st.columns(2)
    col1.image(output, caption="Detected Colonies", use_column_width=True)
    col2.dataframe(df)

    st.subheader("📊 Colony Distribution")
    fig1,ax1 = plt.subplots()
    ax1.pie(
        [colony_info["Small"],colony_info["Medium"],colony_info["Large"]],
        labels=["Small","Medium","Large"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig1)

    X = df[["Area","Perimeter","Circularity"]]
    y = LabelEncoder().fit_transform(df["Health"])
    X = StandardScaler().fit_transform(X)

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)

    svm = SVC(kernel="linear")
    svm.fit(Xtr,ytr)
    svm_acc = accuracy_score(yte, svm.predict(Xte))

    ann = MLPClassifier(hidden_layer_sizes=(25,10), max_iter=1000)
    ann.fit(Xtr,ytr)
    ann_acc = accuracy_score(yte, ann.predict(Xte))

    st.subheader("📈 Model Accuracy")
    st.bar_chart(pd.DataFrame({
        "Accuracy":[svm_acc,ann_acc]
    }, index=["SVM","ANN"]))

    st.subheader("🔍 Confusion Matrix - SVM")
    fig2,ax2 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(yte, svm.predict(Xte), ax=ax2)
    st.pyplot(fig2)

    st.success(f"Total Colonies Detected: {len(df)}")
