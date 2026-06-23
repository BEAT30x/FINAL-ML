import streamlit as st
import tensorflow as tf
import numpy as np
cv2 import cv2
import pickle
import os
from PIL import Image
import tempfile
import time
from collections import deque
import base64
import io

# ============================================
# KONFIGURASI
# ============================================
st.set_page_config(page_title="BISINDO", layout="wide")

# ============================================
# LOAD MODEL - CEK SEMUA LOKASI
# ============================================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Cari file di semua kemungkinan lokasi
    possible_paths = [
        BASE_DIR,
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "model"),
        os.path.join(BASE_DIR, ".."),
    ]
    
    for path in possible_paths:
        image_h5 = os.path.join(path, "image_model.h5")
        video_h5 = os.path.join(path, "video_model.h5")
        image_pkl = os.path.join(path, "image_class_names.pkl")
        video_pkl = os.path.join(path, "video_class_names.pkl")
        
        if os.path.exists(image_h5) and os.path.exists(video_h5):
            try:
                st.info(f"✅ Model ditemukan di: {path}")
                image_model = tf.keras.models.load_model(image_h5)
                video_model = tf.keras.models.load_model(video_h5)
                
                with open(image_pkl, "rb") as f:
                    image_class_names = pickle.load(f)
                with open(video_pkl, "rb") as f:
                    video_class_names = pickle.load(f)
                
                return image_model, video_model, image_class_names, video_class_names
            except Exception as e:
                st.warning(f"⚠️ Gagal load dari {path}: {e}")
                continue
    
    # Tampilkan daftar file yang ada di root
    st.error("❌ Model tidak ditemukan!")
    st.code(f"""
    File di root folder ({BASE_DIR}):
    {os.listdir(BASE_DIR)}
    """)
    return None, None, None, None

image_model, video_model, image_class_names, video_class_names = load_models()

# ============================================
# PREPROCESSING
# ============================================
def preprocess_image(image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_frame_gambar(model, frame, class_names):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    return class_names[pred_class], predictions[0][pred_class] * 100

# ============================================
# HALAMAN UTAMA
# ============================================
st.title("🖐️ BISINDO - Deteksi Gestur")

if image_model is None:
    st.stop()

tab1, tab2 = st.tabs(["📸 Upload Gambar", "📷 Live Camera"])

# ============ TAB 1: UPLOAD ============
with tab1:
    uploaded = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)
        if st.button("Prediksi"):
            processed = preprocess_image(img)
            probs = image_model.predict(processed, verbose=0)[0]
            pred = np.argmax(probs)
            st.success(f"🎯 {image_class_names[pred]} ({probs[pred]*100:.1f}%)")

# ============ TAB 2: LIVE CAMERA ============
with tab2:
    st.write("Aktifkan toggle untuk menyalakan kamera")
    run = st.toggle("▶️ Aktifkan Kamera", key="cam")
    
    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    conf_placeholder = st.empty()
    
    if run:
        cap = None
        # Coba index 0, 1, 2
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
        
        if cap is None or not cap.isOpened():
            st.error("❌ Tidak ada kamera yang terdeteksi!")
            st.info("💡 Pastikan kamera terhubung dan izinkan akses kamera di browser.")
        else:
            st.success("✅ Kamera terdeteksi! Tunjukkan gestur di depan kamera.")
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                label, conf = predict_frame_gambar(image_model, frame, image_class_names)
                
                # Tampilkan hasil
                result_placeholder.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                            padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                    <h1 style="font-size: 3rem; margin: 0;">🎯 {label}</h1>
                    <p style="font-size: 1.2rem;">Confidence: {conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                conf_placeholder.progress(conf/100)
                
                # Tampilkan frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.05)  # Jeda untuk stabilitas
            
            cap.release()
