"""
====================================================================
BISINDO - Deteksi Gestur dengan Kamera (WORKING VERSION)
====================================================================
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import base64
import io
import time

# ============================================
# KONFIGURASI HALAMAN
# ============================================

st.set_page_config(
    page_title="BISINDO - Deteksi Gestur",
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
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.3s ease-in-out;
    }
    .result-box h2 {
        font-size: 3rem;
        margin: 0;
    }
    .result-box .confidence {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .camera-container {
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
    <p>Deteksi Gestur Tangan Bahasa Isyarat Indonesia</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL - PATH DI ROOT (BENAR)
# ============================================

@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # ==========================================
        # MODEL ADA DI ROOT FOLDER
        # ==========================================
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
# FUNGSI PREDIKSI GAMBAR
# ============================================

def predict_image(image, model, class_names):
    """Prediksi gambar dengan preprocessing MobileNetV2"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return class_names[pred_class], confidence, predictions[0]

# ============================================
# HTML KAMERA
# ============================================

def camera_html():
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-container">
            <video id="video" width="100%" height="auto" autoplay style="max-height: 400px; background: #000;"></video>
        </div>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button id="startBtn" style="background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ▶️ Start Camera
            </button>
            <button id="stopBtn" style="background: #dc3545; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
            <button id="captureBtn" style="background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                📸 Capture
            </button>
        </div>
        <br>
        <div id="status" style="font-size: 16px; font-weight: bold; color: #6c757d; min-height: 30px;">
            ⏸️ Klik Start untuk mengaktifkan kamera
        </div>
        <div id="resultPreview" style="font-size: 28px; font-weight: bold; min-height: 60px; padding: 15px; background: #f8f9fa; border-radius: 15px; margin-top: 10px; border: 2px solid #ddd;">
            🎯 Hasil akan muncul di sini
        </div>
        <div id="confidencePreview" style="font-size: 18px; color: #28a745; margin-top: 5px;">
            Confidence: 0%
        </div>
        <canvas id="canvas" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statusDiv = document.getElementById('status');
        const resultPreview = document.getElementById('resultPreview');
        const confidencePreview = document.getElementById('confidencePreview');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const captureBtn = document.getElementById('captureBtn');
        
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        
        function updateUI(label, confidence) {
            if (label && confidence > 0) {
                resultPreview.innerHTML = `<span style="color: #333;">🎯 <b>${label}</b></span>`;
                confidencePreview.textContent = `Confidence: ${confidence.toFixed(1)}%`;
                confidencePreview.style.color = confidence > 80 ? '#28a745' : confidence > 50 ? '#ffc107' : '#dc3545';
                statusDiv.textContent = `✅ Detected: ${label} (${confidence.toFixed(1)}%)`;
                statusDiv.style.color = '#28a745';
            }
        }
        
        function sendToStreamlit(imageData) {
            const data = {
                type: 'camera_frame',
                image: imageData
            };
            window.parent.postMessage(data, '*');
            statusDiv.textContent = '⏳ Memproses...';
            statusDiv.style.color = '#667eea';
        }
        
        function captureFrame() {
            if (!isRunning) {
                statusDiv.textContent = '⚠️ Start camera dulu!';
                statusDiv.style.color = 'orange';
                return;
            }
            
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            sendToStreamlit(imageData);
        }
        
        function startCamera() {
            if (isRunning) {
                statusDiv.textContent = '⚠️ Camera sudah berjalan!';
                statusDiv.style.color = 'orange';
                return;
            }
            
            navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user', width: 640, height: 480 } 
            })
            .then(mediaStream => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.play();
                isRunning = true;
                statusDiv.textContent = '🟢 Camera running...';
                statusDiv.style.color = '#28a745';
                startBtn.disabled = true;
                startBtn.style.opacity = '0.6';
            })
            .catch(err => {
                statusDiv.textContent = '❌ Error: ' + err.message;
                statusDiv.style.color = 'red';
                console.error('Camera error:', err);
            });
        }
        
        function stopCamera() {
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            if (video.srcObject) {
                video.srcObject = null;
            }
            isRunning = false;
            startBtn.disabled = false;
            startBtn.style.opacity = '1';
            statusDiv.textContent = '⏸️ Stopped';
            statusDiv.style.color = '#6c757d';
        }
        
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        captureBtn.addEventListener('click', captureFrame);
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 's' || e.key === 'S') startCamera();
            else if (e.key === 'x' || e.key === 'X') stopCamera();
            else if (e.key === 'c' || e.key === 'C') captureFrame();
        });
        
        window.onload = function() {
            window.parent.postMessage({ type: 'camera_ready' }, '*');
            statusDiv.textContent = '⏸️ Klik Start atau tekan S';
        };
        
        window.addEventListener('beforeunload', function() {
            stopCamera();
        });
        
        window.addEventListener('message', function(event) {
            const data = event.data;
            if (data && data.type === 'prediction_result') {
                updateUI(data.label, data.confidence);
            }
        });
    </script>
    """

# ============================================
# MAIN APP
# ============================================

def main():
    # Load model
    with st.spinner("⏳ Memuat model..."):
        image_model, video_model, image_class_names, video_class_names = load_models()
    
    if image_model is None:
        st.error("❌ Model tidak ditemukan!")
        st.info("📁 Pastikan file model ada di ROOT folder:")
        st.code("""
        /content/
        ├── app.py
        ├── image_model.h5       ← Harus ada!
        ├── video_model.h5       ← Harus ada!
        ├── image_class_names.pkl
        └── video_class_names.pkl
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### 📋 Informasi
        
        **Fitur:**
        - 📷 Kamera Real-Time
        - 📸 Capture Manual
        
        **Shortcuts:**
        - `S` = Start Camera
        - `X` = Stop Camera  
        - `C` = Capture
        
        **Dataset:**
        - 26 Huruf Abjad (A-Z)
        """)
        st.divider()
        st.caption("© 2024 BISINDO Classification")
    
    # Tabs
    tab1, tab2 = st.tabs(["📷 Kamera", "📸 Upload Gambar"])
    
    # ==================== TAB 1: KAMERA ====================
    with tab1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <p style="margin: 0; font-weight: bold;">📷 Deteksi Real-Time</p>
            <p style="margin: 0; font-size: 0.9rem; color: #6c757d;">
                Tunjukkan gestur di depan kamera, lalu klik <kbd>Capture</kbd> atau tekan <kbd>C</kbd>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(camera_html(), height=550)
    
    # ==================== TAB 2: UPLOAD GAMBAR ====================
    with tab2:
        uploaded_file = st.file_uploader(
            "Pilih file gambar...",
            type=["jpg", "jpeg", "png", "bmp", "tiff"]
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="🖼️ Gambar yang diupload", use_container_width=True)
            
            with col2:
                if st.button("🔍 Prediksi", type="primary", use_container_width=True):
                    with st.spinner("⏳ Memproses..."):
                        label, confidence, _ = predict_image(image, image_model, image_class_names)
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h2>🎯 {label}</h2>
                            <p class="confidence">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

# ============================================
# LISTENER UNTUK DATA DARI JAVASCRIPT
# ============================================

query_params = st.query_params

if 'img' in query_params:
    try:
        img_data = query_params['img']
        img_b64 = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        
        image_model, _, image_class_names, _ = load_models()
        
        if image_model is not None:
            label, confidence, _ = predict_image(img, image_model, image_class_names)
            
            st.markdown(f"""
            <script>
                window.parent.postMessage({{
                    type: 'prediction_result',
                    label: '{label}',
                    confidence: {confidence}
                }}, '*');
            </script>
            """, unsafe_allow_html=True)
            
            st.query_params.clear()
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.query_params.clear()

if __name__ == "__main__":
    main()
