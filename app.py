import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.tasks as tasks

# --- Patch to bypass safe globals and load full checkpoint ---
torch.serialization.add_safe_globals([
    DetectionModel,
    torch.nn.modules.container.Sequential,
])

original_torch_safe_load = tasks.torch_safe_load

def patched_torch_safe_load(file):
    # Force weights_only=False to load full checkpoint (unsafe if untrusted)
    return torch.load(file, map_location='cpu', weights_only=False), file

tasks.torch_safe_load = patched_torch_safe_load
# -------------------------------------------------------------

# Load the YOLOv8 model (your trusted model file path)
model_path = r"C:\Users\Muhammed Abdullahi O\Desktop\Corporate Wear Detection System Main File\train2\content\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)

st.title("YOLOv8 Clothing Detection App")

# Sidebar: choose between uploading or webcam
option = st.sidebar.selectbox("Choose Input Method", ["Upload Image", "Use Webcam"])

def detect_and_annotate(image: np.ndarray):
    results = model(image, conf=0.3, iou=0.5)
    annotated_frame = results[0].plot()
    return annotated_frame

# Image Upload Section
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Clothing"):
            annotated_img = detect_and_annotate(img)
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Clothing Items", use_column_width=True)

# Webcam Detection Section
elif option == "Use Webcam":
    if 'webcam_active' not in st.session_state:
        st.session_state['webcam_active'] = False

    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")

    FRAME_WINDOW = st.image([])

    if start_webcam:
        st.session_state['webcam_active'] = True
    if stop_webcam:
        st.session_state['webcam_active'] = False

    cap = cv2.VideoCapture(0)

    while st.session_state['webcam_active']:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        annotated_frame = detect_and_annotate(frame)
        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    cap.release()
