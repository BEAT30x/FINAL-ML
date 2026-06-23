"""
====================================================================
APLIKASI KLASIFIKASI GESTUR TANGAN BISINDO
Bahasa Isyarat Indonesia (BISINDO) - REAL-TIME OTOMATIS
Hasil langsung muncul saat gestur ditampilkan di depan kamera
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
        animation: fadeIn 0.2s ease-in-out;
    }
    .result-box-video {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.2s ease-in-out;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
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
        color: #0f0;
        padding: 8px 25px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        z-index: 10;
        white-space: nowrap;
        border: 2px solid rgba(0,255,0,0.3);
    }
    .overlay-confidence {
        position: absolute;
        bottom: 70px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,0.6);
        color: #fff;
        padding: 4px 15px;
        border-radius: 15px;
        font-size: 13px;
        z-index: 10;
    }
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #ff0000;
        border-radius: 50%;
        animation: pulse 1s infinite;
        margin-right: 5px;
    }
    .confidence-bar-container {
        width: 100%;
        height: 6px;
        background: #e9ecef;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 5px;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.2s ease;
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

def predict_image_from_array(image_array, model, class_names):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.fromarray(image_array)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    return class_names[pred_class], confidence, predictions[0]

# ============================================
# HTML REAL-TIME OTOMATIS
# ============================================

def realtime_auto_html():
    """HTML/JavaScript - Real-time otomatis, hasil langsung muncul"""
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-container">
            <video id="video" width="100%" height="auto" autoplay style="max-height: 450px; background: #000;"></video>
            <div id="overlayLabel" class="overlay-label">🔄 Menunggu gestur...</div>
            <div id="overlayConfidence" class="overlay-confidence">Confidence: 0%</div>
        </div>
        
        <div style="display: flex; justify-content: center; gap: 10px; margin: 15px 0; flex-wrap: wrap;">
            <button id="startBtn" style="background: #28a745; color: white; padding: 10px 30px; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; font-weight: bold;">
                ▶️ Start Auto Detect
            </button>
            <button id="stopBtn" style="background: #dc3545; color: white; padding: 10px 30px; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
            <button id="modeBtn" style="background: #667eea; color: white; padding: 10px 30px; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; font-weight: bold;">
                🔄 Mode: Abjad
            </button>
        </div>
        
        <div id="status" style="font-size: 14px; font-weight: bold; min-height: 25px; color: #6c757d;">
            ⏸️ Klik Start untuk deteksi otomatis
        </div>
        
        <div id="resultDisplay" style="font-size: 28px; font-weight: bold; min-height: 60px; padding: 15px; border-radius: 15px; background: #f8f9fa; border: 2px solid #ddd; margin-top: 10px;">
            <span style="color: #6c757d;">🎯 Hasil deteksi akan muncul di sini</span>
        </div>
        
        <div class="confidence-bar-container">
            <div id="confidenceFill" class="confidence-bar-fill" style="width: 0%; background: linear-gradient(90deg, #28a745, #00d4ff);"></div>
        </div>
        
        <canvas id="canvas" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statusDiv = document.getElementById('status');
        const resultDisplay = document.getElementById('resultDisplay');
        const overlayLabel = document.getElementById('overlayLabel');
        const overlayConfidence = document.getElementById('overlayConfidence');
        const confidenceFill = document.getElementById('confidenceFill');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const modeBtn = document.getElementById('modeBtn');
        
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        let currentMode = 'abjad'; // 'abjad' atau 'kata'
        let lastLabel = 'Menunggu...';
        let lastConfidence = 0;
        let frameCount = 0;
        
        // Update UI dengan hasil prediksi
        function updateResult(label, confidence) {
            lastLabel = label;
            lastConfidence = confidence;
            
            // Update overlay
            overlayLabel.textContent = `🎯 ${label}`;
            overlayConfidence.textContent = `Confidence: ${confidence.toFixed(1)}%`;
            
            // Update result display
            resultDisplay.innerHTML = `<span style="color: #333;">🎯 <b>${label}</b></span>`;
            
            // Update confidence bar
            confidenceFill.style.width = `${Math.min(confidence, 100)}%`;
            
            // Warna berdasarkan confidence
            if (confidence > 80) {
                confidenceFill.style.background = 'linear-gradient(90deg, #28a745, #00d4ff)';
                overlayLabel.style.color = '#0f0';
            } else if (confidence > 50) {
                confidenceFill.style.background = 'linear-gradient(90deg, #ffc107, #ff9800)';
                overlayLabel.style.color = '#ffc107';
            } else {
                confidenceFill.style.background = 'linear-gradient(90deg, #dc3545, #ff6b6b)';
                overlayLabel.style.color = '#ff6b6b';
            }
            
            // Status
            statusDiv.textContent = `✅ Detected: ${label} (${confidence.toFixed(1)}%)`;
            statusDiv.style.color = '#28a745';
        }
        
        // Kirim frame ke Streamlit untuk prediksi
        function sendFrameForPrediction() {
            if (!isRunning) return;
            
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            const data = {
                type: 'camera_frame',
                mode: currentMode,
                image: imageData
            };
            
            // Kirim ke Streamlit
            window.parent.postMessage(data, '*');
        }
        
        // Start auto detection
        function startDetection() {
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
                overlayLabel.style.color = '#0f0';
                resultDisplay.innerHTML = '<span style="color: #667eea;">⏳ Mendeteksi gestur...</span>';
                
                startBtn.textContent = '🔄 Running...';
                startBtn.style.opacity = '0.6';
                startBtn.disabled = true;
                
                // Auto prediction setiap 200ms (5x per detik)
                if (intervalId) clearInterval(intervalId);
                intervalId = setInterval(() => {
                    if (isRunning) {
                        sendFrameForPrediction();
                    }
                }, 200); // 200ms = 5 prediksi per detik
            })
            .catch(err => {
                statusDiv.textContent = '❌ Error: ' + err.message;
                statusDiv.style.color = '#dc3545';
                overlayLabel.textContent = '❌ Camera Error';
                overlayLabel.style.color = '#f00';
                console.error('Camera error:', err);
            });
        }
        
        // Stop detection
        function stopDetection() {
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
            startBtn.textContent = '▶️ Start Auto Detect';
            startBtn.style.opacity = '1';
            startBtn.disabled = false;
            
            statusDiv.textContent = '⏸️ Stopped';
            statusDiv.style.color = '#6c757d';
            overlayLabel.textContent = '⏸️ Stopped';
            overlayLabel.style.color = '#fff';
        }
        
        // Toggle mode
        function toggleMode() {
            if (currentMode === 'abjad') {
                currentMode = 'kata';
                modeBtn.textContent = '🔄 Mode: Kata';
                modeBtn.style.background = '#4facfe';
                statusDiv.textContent = '📌 Mode: Kata (BISINDO Words)';
            } else {
                currentMode = 'abjad';
                modeBtn.textContent = '🔄 Mode: Abjad';
                modeBtn.style.background = '#667eea';
                statusDiv.textContent = '📌 Mode: Abjad (A-Z)';
            }
            statusDiv.style.color = '#667eea';
            
            // Reset hasil
            resultDisplay.innerHTML = `<span style="color: #6c757d;">🔄 Switch ke mode ${currentMode}</span>`;
            overlayLabel.textContent = `🔄 Mode: ${currentMode}`;
        }
        
        // Event listeners
        startBtn.addEventListener('click', startDetection);
        stopBtn.addEventListener('click', stopDetection);
        modeBtn.addEventListener('click', toggleMode);
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 's' || e.key === 'S') {
                startDetection();
            } else if (e.key === 'x' || e.key === 'X') {
                stopDetection();
            } else if (e.key === 'm' || e.key === 'M') {
                toggleMode();
            }
        });
        
        // Menerima hasil prediksi dari Streamlit
        window.addEventListener('message', function(event) {
            const data = event.data;
            if (data && data.type === 'prediction_result') {
                updateResult(data.label, data.confidence);
            }
        });
        
        // Notify ready
        window.onload = function() {
            window.parent.postMessage({ type: 'camera_ready' }, '*');
            statusDiv.textContent = '⏸️ Klik Start atau tekan S untuk memulai deteksi otomatis';
            statusDiv.style.color = '#6c757d';
        };
        
        // Cleanup
        window.addEventListener('beforeunload', function() {
            stopDetection();
        });
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
        - 📷 Auto-Detect (Real-Time)
        
        **Mode Kamera:**
        - **Abjad**: Deteksi huruf (A-Z)
        - **Kata**: Deteksi kata BISINDO
        
        **Keyboard Shortcuts:**
        - `S` = Start Auto-Detect
        - `X` = Stop
        - `M` = Switch Mode
        
        **Dataset:**
        - Abjad: 26 huruf
        - Kata: 24 kata
        """)
        st.divider()
        st.caption("© 2024 BISINDO Classification")
    
    tab1, tab2, tab3 = st.tabs([
        "📸 Gambar", "🎬 Video", "📷 Auto-Detect"
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
                    label, conf, _ = predict_image_from_array(np.array(image), image_model, image_class_names)
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
                # Simpan video ke session
                st.session_state.uploaded_video = uploaded
                st.success("✅ Video diupload! Gunakan fitur Auto-Detect untuk prediksi.")
    
    # ==================== TAB 3: AUTO-DETECT ====================
    with tab3:
        st.markdown("""
        <div class="card">
            <h3>📷 Auto-Detect <span style="display:inline-block; background:#ff0000; color:white; padding:0 10px; border-radius:10px; font-size:12px; animation:pulse 1s infinite;">LIVE</span></h3>
            <p style="font-size:16px; font-weight:bold; color:#28a745;">
                ⚡ Tunjukkan gestur di depan kamera, hasil langsung muncul!
            </p>
            <p style="color: red; font-size: 0.9rem;">⚠️ Izinkan akses kamera saat diminta browser</p>
            <p style="font-size: 0.9rem;">
                💡 <b>Shortcuts:</b> <kbd>S</kbd> Start | <kbd>X</kbd> Stop | <kbd>M</kbd> Ganti Mode
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-detect camera
        st.components.v1.html(realtime_auto_html(), height=650)

# ============================================
# LISTENER UNTUK DATA DARI JAVASCRIPT
# ============================================

# Proses data dari query params
query_params = st.query_params

if 'camera_image' in query_params:
    try:
        image_data = query_params['camera_image']
        mode = query_params.get('camera_mode', 'abjad')
        
        # Decode image
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        # Load models
        image_model, video_model, image_class_names, video_class_names = load_models()
        
        if image_model is not None:
            if mode == 'kata':
                # Untuk kata, gunakan video model
                # Simpan ke session state
                st.session_state.kata_frame = img_array
                label = "Kata: Gunakan video"
                confidence = 0
            else:
                # Abjad mode - prediksi langsung
                label, confidence, _ = predict_image_from_array(img_array, image_model, image_class_names)
                
                # Kirim hasil balik ke JavaScript melalui meta tag
                st.session_state.prediction_label = label
                st.session_state.prediction_confidence = confidence
                
                # Inject JavaScript untuk update hasil di frontend
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
            st.rerun()
            
    except Exception as e:
        print(f"Error processing camera: {e}")

if __name__ == "__main__":
    main()
