"""
====================================================================
APLIKASI KLASIFIKASI GESTUR TANGAN BISINDO
Bahasa Isyarat Indonesia (BISINDO) - FINAL STABLE
====================================================================
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import tempfile
import time

# ============================================
# KONFIGURASI HALAMAN
# ============================================

st.set_page_config(
    page_title="BISINDO - Bahasa Isyarat Indonesia",
    page_icon="🖐️",
    layout="wide"
)

# ============================================
# CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box-video {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🖐️ BISINDO</h1>
    <p>Klasifikasi Gestur Tangan Bahasa Isyarat Indonesia</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        image_model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "image_model.h5")
        )

        video_model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "video_model.h5")
        )

        with open(
            os.path.join(BASE_DIR, "image_class_names.pkl"),
            "rb"
        ) as f:
            image_class_names = pickle.load(f)

        with open(
            os.path.join(BASE_DIR, "video_class_names.pkl"),
            "rb"
        ) as f:
            video_class_names = pickle.load(f)

        return (
            image_model,
            video_model,
            image_class_names,
            video_class_names
        )

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None

# ============================================
# FUNGSI PREDIKSI
# ============================================

def predict_image(model, image, class_names):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    return class_names[pred_class], confidence, predictions[0]

def predict_video(model, video_file, class_names):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    frames = []
    
    if not cap.isOpened():
        os.remove(tmp_path)
        return None, 0, None
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    os.remove(tmp_path)
    
    total_frames = len(all_frames)
    if total_frames == 0:
        return None, 0, None
    
    indices = np.linspace(0, total_frames-1, 20, dtype=int)
    for idx in indices:
        if idx < total_frames:
            frame = all_frames[idx]
            frame = cv2.resize(frame, (96, 96))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        else:
            frames.append(np.zeros((96, 96, 3), dtype=np.float32))
    
    video_array = np.array(frames, dtype=np.float32)
    video_array = np.expand_dims(video_array, axis=0)
    predictions = model.predict(video_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    return class_names[pred_class], confidence, predictions[0]

def predict_frame_gambar(model, frame, class_names):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    return class_names[pred_class], confidence, predictions[0]

# ============================================
# MAIN
# ============================================

def main():
    image_model, video_model, image_class_names, video_class_names = load_models()
    
    if image_model is None:
        st.error("❌ Gagal load model!")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Gambar", "🎬 Video", "📷 Kamera Abjad", "📷 Kamera Kata"
    ])
    
    # ==================== TAB 1: GAMBAR ====================
    with tab1:
        uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_container_width=True)
            with col2:
                if st.button("Prediksi", type="primary"):
                    label, conf, _ = predict_image(image_model, image, image_class_names)
                    st.markdown(f"""
                    <div class="result-box">
                        <h2>🎯 {label}</h2>
                        <p>Confidence: {conf:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 2: VIDEO ====================
    with tab2:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        if uploaded:
            if st.button("Prediksi Video", type="primary"):
                label, conf, _ = predict_video(video_model, uploaded, video_class_names)
                if label:
                    st.markdown(f"""
                    <div class="result-box-video">
                        <h2>🎯 {label}</h2>
                        <p>Confidence: {conf:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Gagal memproses video!")
    
    # ==================== TAB 3: KAMERA ABJAD ====================
    with tab3:
        st.markdown("### 📷 Kamera Abjad")
        
        image_display = st.empty()
        result_display = st.empty()
        
        col_start, col_stop = st.columns(2)
        with col_start:
            start = st.button("▶️ Start", type="primary", use_container_width=True)
        with col_stop:
            stop = st.button("⏹️ Stop", use_container_width=True)
        
        if 'cam_abjad' not in st.session_state:
            st.session_state.cam_abjad = False
            st.session_state.cam_result = "Menunggu..."
            st.session_state.cam_conf = 0
        
        if start:
            st.session_state.cam_abjad = True
        
        if stop:
            st.session_state.cam_abjad = False
            image_display.empty()
            result_display.empty()
            st.rerun()
        
        if st.session_state.cam_abjad:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    while st.session_state.cam_abjad:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        label, conf, _ = predict_frame_gambar(image_model, frame, image_class_names)
                        st.session_state.cam_result = label
                        st.session_state.cam_conf = conf
                        
                        cv2.putText(frame, f"{label} ({conf:.1f}%)", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image_display.image(frame_rgb, channels="RGB", use_container_width=True)
                        except:
                            pass
                        
                        try:
                            result_display.markdown(f"""
                            <div class="result-box">
                                <h2>🎯 {label}</h2>
                                <p>Confidence: {conf:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        except:
                            pass
                        
                        time.sleep(0.05)
                    
                    cap.release()
                    st.session_state.cam_abjad = False
                else:
                    st.error("❌ Kamera tidak tersedia!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.cam_abjad = False
    
    # ==================== TAB 4: KAMERA KATA ====================
    with tab4:
        st.markdown("### 📷 Kamera Kata")
        
        image_display_kata = st.empty()
        result_display_kata = st.empty()
        
        col_start_k, col_stop_k = st.columns(2)
        with col_start_k:
            start_k = st.button("▶️ Start", type="primary", use_container_width=True, key="start_kata")
        with col_stop_k:
            stop_k = st.button("⏹️ Stop", use_container_width=True, key="stop_kata")
        
        if 'cam_kata' not in st.session_state:
            st.session_state.cam_kata = False
            st.session_state.cam_kata_result = "Mengisi buffer..."
            st.session_state.cam_kata_conf = 0
            st.session_state.buffer_kata = []
        
        if start_k:
            st.session_state.cam_kata = True
            st.session_state.buffer_kata = []
        
        if stop_k:
            st.session_state.cam_kata = False
            st.session_state.buffer_kata = []
            image_display_kata.empty()
            result_display_kata.empty()
            st.rerun()
        
        if st.session_state.cam_kata:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    while st.session_state.cam_kata:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Proses buffer
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (96, 96))
                        frame_norm = frame_resized.astype(np.float32) / 255.0
                        st.session_state.buffer_kata.append(frame_norm)
                        
                        if len(st.session_state.buffer_kata) > 20:
                            st.session_state.buffer_kata = st.session_state.buffer_kata[-20:]
                        
                        label = "Mengisi buffer..."
                        conf = 0
                        if len(st.session_state.buffer_kata) >= 20:
                            frames = np.array(st.session_state.buffer_kata[-20:])
                            frames = np.expand_dims(frames, axis=0)
                            predictions = video_model.predict(frames, verbose=0)
                            pred_class = np.argmax(predictions[0])
                            conf = np.max(predictions[0]) * 100
                            label = video_class_names[pred_class]
                            st.session_state.cam_kata_result = label
                            st.session_state.cam_kata_conf = conf
                        else:
                            st.session_state.cam_kata_result = f"Mengisi buffer... ({len(st.session_state.buffer_kata)}/20)"
                            st.session_state.cam_kata_conf = 0
                        
                        cv2.putText(frame, f"{label} ({conf:.1f}%)", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Buffer: {len(st.session_state.buffer_kata)}/20", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image_display_kata.image(frame_rgb, channels="RGB", use_container_width=True)
                        except:
                            pass
                        
                        try:
                            if conf > 0:
                                result_display_kata.markdown(f"""
                                <div class="result-box-video">
                                    <h2>🎯 {label}</h2>
                                    <p>Confidence: {conf:.1f}%</p>
                                    <p>Buffer: {len(st.session_state.buffer_kata)}/20 ✅</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                result_display_kata.markdown(f"""
                                <div style="background:#6c757d; padding:1.5rem; border-radius:15px; color:white; text-align:center;">
                                    <h2>⏳ {label}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        except:
                            pass
                        
                        time.sleep(0.05)
                    
                    cap.release()
                    st.session_state.cam_kata = False
                else:
                    st.error("❌ Kamera tidak tersedia!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.cam_kata = False

if __name__ == "__main__":
    main()
