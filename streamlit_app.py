import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

st.title("👤 Face Shape & Hairstyle Recommender 💇‍♂️💇‍♀️")
st.write("Upload a photo to detect your face shape and get hairstyle suggestions!")

uploaded_file = st.file_uploader("📷 Upload a front-facing image", type=["jpg", "jpeg", "png"])

# Face shape detection using MediaPipe
def detect_face_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    if not results.multi_face_landmarks:
        return "Face not detected"

    # Dummy logic: you can improve it later
    landmarks = results.multi_face_landmarks[0]
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]

    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    ratio = width / height

    if ratio > 0.9:
        return "Round"
    elif ratio < 0.75:
        return "Oval"
    else:
        return "Square"

def recommend_hairstyle(face_shape):
    styles = {
        "Round": "Try long layered cuts or high-volume styles to elongate your face.",
        "Oval": "You’re lucky! Almost any hairstyle suits you.",
        "Square": "Soft curls or layered styles work best to soften angles.",
    }
    return styles.get(face_shape, "No recommendation found.")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    face_shape = detect_face_shape(image)
    st.subheader(f"Detected Face Shape: {face_shape}")
    recommendation = recommend_hairstyle(face_shape)
    st.write(recommendation)
