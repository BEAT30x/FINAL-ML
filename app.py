"""
====================================================================
APLIKASI KLASIFIKASI GESTUR TANGAN BISINDO
Bahasa Isyarat Indonesia (BISINDO) - WITH CAMERA SUPPORT
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
import base64
import io

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
    .camera-preview {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        background: #000;
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

# ============================================
# FUNGSI CAMERA CAPTURE (JAVASCRIPT)
# ============================================

def camera_capture_html():
    """HTML/JavaScript untuk capture dari kamera browser"""
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-preview">
            <video id="video" width="100%" height="auto" autoplay style="max-height: 400px;"></video>
        </div>
        <br>
        <button id="capture" style="background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
            📸 Capture & Predict
        </button>
        <button id="capture_kata" style="background: #4facfe; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold; margin-left: 10px;">
            🎬 Capture Kata
        </button>
        <canvas id="canvas" style="display: none;"></canvas>
        <br><br>
        <div id="result" style="font-size: 18px; font-weight: bold; color: #333; min-height: 50px;"></div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const captureBtn = document.getElementById('capture');
        const captureKataBtn = document.getElementById('capture_kata');
        const resultDiv = document.getElementById('result');
        
        // Inisialisasi kamera
        let stream = null;
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } })
            .then(mediaStream => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                resultDiv.innerHTML = '🟢 Camera ready!';
                resultDiv.style.color = 'green';
            })
            .catch(err => {
                resultDiv.innerHTML = '❌ Gagal mengakses kamera: ' + err.message;
                resultDiv.style.color = 'red';
            });
        
        // Fungsi capture
        function captureImage(mode) {
            if (!stream) {
                resultDiv.innerHTML = '❌ Kamera belum siap!';
                resultDiv.style.color = 'red';
                return;
            }
            
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Kirim ke Streamlit
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            const data = {
                type: 'camera_image',
                mode: mode,
                image: imageData
            };
            
            resultDiv.innerHTML = '⏳ Memproses...';
            resultDiv.style.color = '#667eea';
            
            // Kirim data ke Streamlit
            window.parent.postMessage(data, '*');
        }
        
        captureBtn.addEventListener('click', function() {
            captureImage('abjad');
        });
        
        captureKataBtn.addEventListener('click', function() {
            captureImage('kata');
        });
        
        // Kirim notifikasi ke Streamlit bahwa JavaScript siap
        window.onload = function() {
            window.parent.postMessage({ type: 'camera_ready' }, '*');
        };
    </script>
    """

# ============================================
# MAIN
# ============================================

def main():
    image_model, video_model, image_class_names, video_class_names = load_models()
    
    if image_model is None:
        st.error("❌ Gagal load model!")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### 📋 Informasi
        
        **Fitur:**
        - 📸 Gambar (Abjad A-Z)
        - 🎬 Video (Kata BISINDO)
        - 📷 Kamera (Abjad & Kata)
        
        **Dataset:**
        - Abjad: 26 huruf
        - Kata: 24 kata
        """)
        st.divider()
        st.caption("© 2024 BISINDO Classification")
    
    tab1, tab2, tab3 = st.tabs([
        "📸 Gambar", "🎬 Video", "📷 Kamera"
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
    
    # ==================== TAB 3: KAMERA ====================
    with tab3:
        st.markdown("""
        <div class="card">
            <h3>📷 Kamera BISINDO</h3>
            <p>Ambil gambar dari kamera untuk mendeteksi gestur</p>
            <p style="color: red; font-size: 0.9rem;">⚠️ Izinkan akses kamera saat diminta browser</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_cam, col_result = st.columns([2, 1])
        
        with col_cam:
            # HTML Kamera
            st.components.v1.html(camera_capture_html(), height=550)
            
            # Hidden component untuk menerima data dari JavaScript
            camera_data = st.session_state.get('camera_data', None)
        
        with col_result:
            st.markdown("### 🎯 Hasil Deteksi")
            result_container = st.empty()
            
            # Upload alternatif jika JavaScript tidak berfungsi
            st.markdown("---")
            st.markdown("### 📤 Alternatif: Upload Hasil Capture")
            
            captured_file = st.file_uploader(
                "Upload gambar hasil capture...",
                type=["jpg", "jpeg", "png"],
                key="captured_image"
            )
            
            if captured_file is not None:
                image = Image.open(captured_file)
                st.image(image, caption="📷 Gambar dari kamera", use_container_width=True)
                
                # Pilih mode
                mode = st.radio("Pilih mode:", ["Abjad", "Kata"], horizontal=True)
                
                if st.button("🔍 Prediksi Hasil Kamera", type="primary", use_container_width=True):
                    with st.spinner("⏳ Memproses prediksi..."):
                        if mode == "Abjad":
                            label, conf, all_probs = predict_image(
                                image_model, image, image_class_names
                            )
                            class_names = image_class_names
                        else:
                            # Untuk kata, kita butuh video. Tapi kita bisa menggunakan gambar sebagai sequence
                            st.warning("⚠️ Mode Kata membutuhkan video. Gunakan fitur Capture Kata di atas.")
                            label = "Gunakan Capture Kata"
                            conf = 0
                            all_probs = None
                        
                        if all_probs is not None:
                            result_container.markdown(f"""
                            <div class="result-box fade-in">
                                <h2>🎯 {label}</h2>
                                <p class="confidence">Confidence: {conf:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("📊 Top 5 Prediksi")
                            top_indices = np.argsort(all_probs)[-5:][::-1]
                            for idx in top_indices:
                                prob = float(all_probs[idx] * 100)
                                col_progress, col_label = st.columns([3, 1])
                                with col_progress:
                                    st.progress(prob / 100)
                                with col_label:
                                    st.write(f"{class_names[idx]} {prob:.1f}%")

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()
