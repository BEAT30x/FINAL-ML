"""
====================================================================
BISINDO - DETEKSI GESTUR REAL-TIME
Hasil langsung muncul saat gestur di depan kamera
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
        animation: fadeIn 0.3s ease-in-out;
        margin-top: 1rem;
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
        position: relative;
    }
    .overlay-label {
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,0.8);
        color: #00ff00;
        padding: 8px 25px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        z-index: 10;
        border: 2px solid rgba(0,255,0,0.3);
        text-align: center;
        width: 80%;
        max-width: 400px;
    }
    .live-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #ff0000;
        border-radius: 50%;
        animation: pulse 1s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
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
# LOAD MODEL
# ============================================

@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        image_model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "image_model.h5")
        )

        with open(
            os.path.join(BASE_DIR, "image_class_names.pkl"),
            "rb"
        ) as f:
            image_class_names = pickle.load(f)

        return image_model, image_class_names

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

# ============================================
# FUNGSI PREDIKSI
# ============================================

def predict_image_from_bytes(image_bytes, model, class_names):
    """Prediksi dari bytes image"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return class_names[pred_class], confidence

# ============================================
# HTML KAMERA - AUTO DETECT
# ============================================

def camera_auto_html():
    """HTML dengan auto-capture setiap 500ms"""
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-container" style="position: relative;">
            <video id="video" width="100%" height="auto" autoplay style="max-height: 400px; background: #000;"></video>
            <div id="overlayLabel" class="overlay-label" style="color: #ffcc00;">
                🔄 Menunggu gestur...
            </div>
        </div>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button id="startBtn" style="background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                <span class="live-dot" style="display:inline-block; width:10px; height:10px; background:white; border-radius:50%; margin-right:5px;"></span>
                ▶️ Start Auto-Detect
            </button>
            <button id="stopBtn" style="background: #dc3545; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
        </div>
        <br>
        <div id="status" style="font-size: 14px; font-weight: bold; color: #6c757d; min-height: 25px;">
            ⏸️ Klik Start untuk deteksi otomatis
        </div>
        <div id="resultDisplay" style="font-size: 32px; font-weight: bold; min-height: 60px; padding: 15px; background: #f8f9fa; border-radius: 15px; margin-top: 10px; border: 2px solid #ddd;">
            <span style="color: #6c757d;">🎯 Hasil akan muncul di sini</span>
        </div>
        <div id="confidenceDisplay" style="font-size: 18px; margin-top: 5px; color: #28a745;">
            Confidence: 0%
        </div>
        <canvas id="canvas" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statusDiv = document.getElementById('status');
        const resultDisplay = document.getElementById('resultDisplay');
        const confidenceDisplay = document.getElementById('confidenceDisplay');
        const overlayLabel = document.getElementById('overlayLabel');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        
        // Fungsi update UI
        function updateUI(label, confidence) {
            const labelText = label || 'Menunggu...';
            const confText = confidence || 0;
            
            // Update result display
            resultDisplay.innerHTML = `<span style="color: ${confText > 70 ? '#28a745' : confText > 40 ? '#ffc107' : '#dc3545'};">🎯 <b>${labelText}</b></span>`;
            confidenceDisplay.textContent = `Confidence: ${confText.toFixed(1)}%`;
            confidenceDisplay.style.color = confText > 70 ? '#28a745' : confText > 40 ? '#ffc107' : '#dc3545';
            
            // Update overlay
            overlayLabel.textContent = `🎯 ${labelText} (${confText.toFixed(1)}%)`;
            overlayLabel.style.color = confText > 70 ? '#00ff00' : confText > 40 ? '#ffcc00' : '#ff6b6b';
            
            // Update status
            if (label && confText > 0) {
                statusDiv.textContent = `✅ Detected: ${labelText}`;
                statusDiv.style.color = '#28a745';
            }
        }
        
        // Kirim frame ke Streamlit
        function sendFrame() {
            if (!isRunning) return;
            
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            
            // Kirim ke Streamlit via URL
            const currentUrl = window.location.href.split('?')[0];
            window.location.href = currentUrl + '?img=' + encodeURIComponent(imageData) + '&t=' + Date.now();
            
            statusDiv.textContent = '⏳ Memproses...';
            statusDiv.style.color = '#667eea';
        }
        
        // Start camera
        function startCamera() {
            if (isRunning) {
                statusDiv.textContent = '⚠️ Already running!';
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
                
                statusDiv.textContent = '🟢 Auto-detection running...';
                statusDiv.style.color = '#28a745';
                overlayLabel.textContent = '🔄 Mendeteksi...';
                overlayLabel.style.color = '#ffcc00';
                resultDisplay.innerHTML = '<span style="color: #667eea;">⏳ Mendeteksi gestur...</span>';
                
                startBtn.textContent = '🔄 Running...';
                startBtn.style.opacity = '0.6';
                startBtn.disabled = true;
                
                // Auto capture every 500ms
                if (intervalId) clearInterval(intervalId);
                intervalId = setInterval(sendFrame, 500);
            })
            .catch(err => {
                statusDiv.textContent = '❌ Error: ' + err.message;
                statusDiv.style.color = 'red';
                overlayLabel.textContent = '❌ Camera Error';
                overlayLabel.style.color = 'red';
                console.error('Camera error:', err);
            });
        }
        
        // Stop camera
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
            
            startBtn.textContent = '▶️ Start Auto-Detect';
            startBtn.style.opacity = '1';
            startBtn.disabled = false;
            
            statusDiv.textContent = '⏸️ Stopped';
            statusDiv.style.color = '#6c757d';
            overlayLabel.textContent = '⏸️ Stopped';
            overlayLabel.style.color = '#ffffff';
        }
        
        // Event listeners
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 's' || e.key === 'S') startCamera();
            else if (e.key === 'x' || e.key === 'X') stopCamera();
        });
        
        // Cleanup
        window.addEventListener('beforeunload', function() {
            stopCamera();
        });
        
        // Menerima hasil dari Streamlit
        window.addEventListener('message', function(event) {
            const data = event.data;
            if (data && data.type === 'prediction_result') {
                updateUI(data.label, data.confidence);
            }
        });
        
        // Notify ready
        window.onload = function() {
            statusDiv.textContent = '⏸️ Klik Start atau tekan S untuk deteksi otomatis';
            statusDiv.style.color = '#6c757d';
        };
    </script>
    """

# ============================================
# MAIN APP
# ============================================

def main():
    # Load model
    with st.spinner("⏳ Memuat model..."):
        model, class_names = load_models()
    
    if model is None:
        st.error("❌ Model tidak ditemukan!")
        st.info("📁 Pastikan file model ada di ROOT folder:")
        st.code("""
        /content/
        ├── app.py
        ├── image_model.h5       ← Harus ada!
        ├── image_class_names.pkl
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### 📋 Informasi
        
        **Fitur:**
        - 📷 Auto-Detect (Real-Time)
        - 📸 Upload Gambar
        
        **Shortcuts:**
        - `S` = Start
        - `X` = Stop
        
        **Dataset:**
        - 26 Huruf Abjad (A-Z)
        
        **Cara Pakai:**
        1. Klik Start
        2. Tunjukkan gestur di depan kamera
        3. Hasil langsung muncul otomatis!
        """)
        st.divider()
        st.caption("© 2024 BISINDO Classification")
    
    # Tabs
    tab1, tab2 = st.tabs(["📷 Auto-Detect", "📸 Upload Gambar"])
    
    # ==================== TAB 1: AUTO-DETECT ====================
    with tab1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <p style="margin: 0; font-weight: bold;">📷 Auto-Detect <span style="display:inline-block; background:#ff0000; color:white; padding:0 10px; border-radius:10px; font-size:12px;">LIVE</span></p>
            <p style="margin: 0; font-size: 0.9rem; color: #6c757d;">
                ⚡ Tunjukkan gestur di depan kamera, <b>hasil langsung muncul otomatis</b> tanpa klik!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder untuk hasil
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        # Kamera HTML
        st.components.v1.html(camera_auto_html(), height=600)
    
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
                        label, confidence = predict_image_from_bytes(
                            uploaded_file.getvalue(), model, class_names
                        )
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h2>🎯 {label}</h2>
                            <p class="confidence">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

# ============================================
# LISTENER UNTUK DATA DARI JAVASCRIPT
# ============================================

# Proses data dari query params
query_params = st.query_params

if 'img' in query_params:
    try:
        img_data = query_params['img']
        
        # Handle jika ada timestamp
        if '&t=' in img_data:
            img_data = img_data.split('&t=')[0]
        
        img_b64 = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        
        # Load model
        model, class_names = load_models()
        
        if model is not None:
            label, confidence = predict_image_from_bytes(img_bytes, model, class_names)
            
            # Kirim hasil ke JavaScript
            st.markdown(f"""
            <script>
                window.parent.postMessage({{
                    type: 'prediction_result',
                    label: '{label}',
                    confidence: {confidence}
                }}, '*');
            </script>
            """, unsafe_allow_html=True)
            
            # Clear query params
            st.query_params.clear()
            
    except Exception as e:
        print(f"Error: {e}")
        st.query_params.clear()

if __name__ == "__main__":
    main()
