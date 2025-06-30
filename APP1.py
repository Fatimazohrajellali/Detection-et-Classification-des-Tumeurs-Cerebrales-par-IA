import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Charger les modèles (attention au chemin)
binary_model = load_model(r"C:\Users\INKA\Desktop\tumor_app\CNN_model.h5", compile=False)
multiclass_model = load_model(r"C:\Users\INKA\Desktop\tumor_app\efficientnet_brain_tumor_modelMULTIPLE.h5", compile=False)

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Fonction de prétraitement pour modèle binaire avec OpenCV
def preprocess_binary(img_pil, target_size=(224, 224)):
    # Convertir PIL en array OpenCV (BGR)
    img = np.array(img_pil.convert('RGB'))  # RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convertir en BGR pour OpenCV

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Égalisation simple de l'histogramme
    gray = cv2.equalizeHist(gray)

    # Seuillage automatique (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Nettoyage : fermeture morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Trouver contours pour recadrage
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = gray[y:y+h, x:x+w]
    else:
        cropped = gray

    # Redimensionner
    resized = cv2.resize(cropped, target_size)

    # Normaliser
    normalized = resized.astype(np.float32) / 255.0

    # Ajouter dimensions batch et canal
    input_arr = np.expand_dims(normalized, axis=(0, -1))  # (1, H, W, 1)

    return input_arr

# Prétraitement pour multiclass EfficientNet (RGB + preprocess_input)
def preprocess_multi(img_pil, target_size=(224, 224)):
    img = img_pil.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

st.title("🧠 Détection et Classification de Tumeurs Cérébrales")

uploaded_file = st.file_uploader("📤 Télécharger une image IRM", type=["jpg", "jpeg", "png"])

if "detect_result" not in st.session_state:
    st.session_state.detect_result = ""
if "classify_result" not in st.session_state:
    st.session_state.classify_result = ""

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", width=150)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Détection (Présence de tumeur)"):
            input_img = preprocess_binary(image)
            pred = binary_model.predict(input_img)[0]
            label = "Tumeur détectée" if np.argmax(pred) == 1 else "Aucune tumeur détectée"
            st.session_state.detect_result = label

        if st.session_state.detect_result:
            st.success(f"Résultat détection : {st.session_state.detect_result}")

    with col2:
        if st.button("🧬 Classification (Type de tumeur)"):
            input_img = preprocess_multi(image)
            pred = multiclass_model.predict(input_img)[0]
            class_idx = np.argmax(pred)
            st.session_state.classify_result = f"Type : {class_labels[class_idx]}"

        if st.session_state.classify_result:
            st.success(f"Résultat classification : {st.session_state.classify_result}")
