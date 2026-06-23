"""
====================================================================
APLIKASI KLASIFIKASI GESTUR TANGAN BISINDO
Bahasa Isyarat Indonesia (BISINDO) - REAL-TIME PREDIKSI LANGSUNG
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
        animation: fadeIn 0.3s ease-in-out;
    }
    .result-box-video {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.3s ease-in-out;
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
        position: relative;
    }
    .overlay-label {
        position: absolute;
        bottom: 10px;
        left: 10px;
        background: rgba(0,0,0,0.7);
        color: #0f0;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
    }
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .live-badge {
        display: inline-block;
        background: #ff0000;
        color: white;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 12px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
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
# REAL-TIME CAMERA HTML (DENGAN RESULT DISPLAY)
# ============================================

def realtime_camera_html():
    """HTML/JavaScript untuk real-time camera dengan display hasil langsung"""
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-preview" style="position: relative;">
            <video id="video" width="100%" height="auto" autoplay style="max-height: 400px; background: #000;"></video>
            <div id="overlayLabel" class="overlay-label">⏳ Menunggu...</div>
        </div>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button id="startBtn" style="background: #28a745; color: white; padding: 10px 25px; border: none; border-radius: 10px; font-size: 14px; cursor: pointer; font-weight: bold;">
                ▶️ Start
            </button>
            <button id="stopBtn" style="background: #dc3545; color: white; padding: 10px 25px; border: none; border-radius: 10px; font-size: 14px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
            <button id="captureBtn" style="background: #667eea; color: white; padding: 10px 25px; border: none; border-radius: 10px; font-size: 14px; cursor: pointer; font-weight: bold;">
                📸 Capture
            </button>
            <button id="kataBtn" style="background: #4facfe; color: white; padding: 10px 25px; border: none; border-radius: 10px; font-size: 14px; cursor: pointer; font-weight: bold;">
                🎬 Kata Mode
            </button>
        </div>
        <br>
        <div id="status" style="font-size: 14px; font-weight: bold; min-height: 25px; color: #6c757d;">⏸️ Click Start or press S</div>
        <div id="resultDisplay" style="font-size: 24px; font-weight: bold; min-height: 50px; margin-top: 10px; padding: 10px; border-radius: 10px; background: #f8f9fa; border: 2px solid #ddd;">
            <span style="color: #6c757d;">🎯 Hasil akan muncul di sini</span>
        </div>
        <div id="confidenceBar" style="margin-top: 5px; width: 100%; height: 8px; background: #e9ecef; border-radius: 5px; overflow: hidden;">
            <div id="confidenceFill" style="height: 100%; width: 0%; background: linear-gradient(90deg, #28a745, #007bff); border-radius: 5px; transition: width 0.3s;"></div>
        </div>
        <canvas id="canvas" style="display: none;"></canvas>
        <canvas id="canvasKata" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const canvasKata = document.getElementById('canvasKata');
        const ctx = canvas.getContext('2d');
        const ctxKata = canvasKata.getContext('2d');
        const statusDiv = document.getElementById('status');
        const resultDisplay = document.getElementById('resultDisplay');
        const overlayLabel = document.getElementById('overlayLabel');
        const confidenceBar = document.getElementById('confidenceFill');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const captureBtn = document.getElementById('captureBtn');
        const kataBtn = document.getElementById('kataBtn');
        
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        let currentMode = 'abjad';
        let lastPrediction = 'Menunggu...';
        let lastConfidence = 0;
        
        // Update UI dengan hasil
        function updateResult(label, confidence) {
            lastPrediction = label;
            lastConfidence = confidence;
            
            // Update display
            resultDisplay.innerHTML = `<span style="color: #333;">🎯 <b>${label}</b></span>`;
            overlayLabel.textContent = `${label} (${confidence.toFixed(1)}%)`;
            
            // Update confidence bar
            confidenceBar.style.width = `${Math.min(confidence, 100)}%`;
            
            // Warna berdasarkan confidence
            if (confidence > 80) {
                confidenceBar.style.background = 'linear-gradient(90deg, #28a745, #00d4ff)';
            } else if (confidence > 50) {
                confidenceBar.style.background = 'linear-gradient(90deg, #ffc107, #ff9800)';
            } else {
                confidenceBar.style.background = 'linear-gradient(90deg, #dc3545, #ff6b6b)';
            }
        }
        
        // Kirim ke Streamlit
        function sendToStreamlit(imageData, mode) {
            const data = {
                type: 'camera_frame',
                mode: mode,
                image: imageData
            };
            window.parent.postMessage(data, '*');
            
            // Update sementara
            statusDiv.textContent = '⏳ Memproses...';
            statusDiv.style.color = '#667eea';
            resultDisplay.innerHTML = `<span style="color: #667eea;">⏳ Memproses...</span>`;
        }
        
        // Capture frame
        function captureFrame(mode) {
            if (!isRunning) {
                statusDiv.textContent = '⚠️ Start camera dulu!';
                statusDiv.style.color = 'orange';
                return;
            }
            
            const targetCanvas = mode === 'kata' ? canvasKata : canvas;
            const targetCtx = mode === 'kata' ? ctxKata : ctx;
            
            targetCanvas.width = video.videoWidth || 640;
            targetCanvas.height = video.videoHeight || 480;
            targetCtx.drawImage(video, 0, 0, targetCanvas.width, targetCanvas.height);
            
            const imageData = targetCanvas.toDataURL('image/jpeg', 0.8);
            sendToStreamlit(imageData, mode);
        }
        
        // Start camera dengan real-time
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
                statusDiv.textContent = '🟢 Running (Real-Time)';
                statusDiv.style.color = '#28a745';
                overlayLabel.textContent = '🟢 Active';
                overlayLabel.style.color = '#0f0';
                
                // REAL-TIME: Prediksi setiap 300ms
                if (intervalId) clearInterval(intervalId);
                intervalId = setInterval(() => {
                    if (isRunning) {
                        // Ambil frame untuk real-time
                        canvas.width = video.videoWidth || 640;
                        canvas.height = video.videoHeight || 480;
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const imageData = canvas.toDataURL('image/jpeg', 0.7);
                        
                        // Kirim untuk prediksi real-time (mode abjad otomatis)
                        const data = {
                            type: 'camera_frame',
                            mode: 'realtime',
                            image: imageData
                        };
                        window.parent.postMessage(data, '*');
                    }
                }, 300); // 300ms = ~3 prediksi per detik
                
                // Update button states
                startBtn.style.opacity = '0.5';
                startBtn.disabled = true;
            })
            .catch(err => {
                statusDiv.textContent = '❌ Error: ' + err.message;
                statusDiv.style.color = '#dc3545';
                overlayLabel.textContent = '❌ Error';
                overlayLabel.style.color = '#f00';
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
            statusDiv.textContent = '⏸️ Stopped';
            statusDiv.style.color = '#6c757d';
            overlayLabel.textContent = '⏸️ Stopped';
            overlayLabel.style.color = '#fff';
            startBtn.style.opacity = '1';
            startBtn.disabled = false;
        }
        
        // Event listeners
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        captureBtn.addEventListener('click', function() {
            captureFrame('abjad');
        });
        kataBtn.addEventListener('click', function() {
            captureFrame('kata');
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 's' || e.key === 'S') {
                startCamera();
            } else if (e.key === 'x' || e.key === 'X') {
                stopCamera();
            } else if (e.key === 'c' || e.key === 'C') {
                captureFrame('abjad');
            } else if (e.key === 'k' || e.key === 'K') {
                captureFrame('kata');
            }
        });
        
        // Notify ready
        window.onload = function() {
            window.parent.postMessage({ type: 'camera_ready' }, '*');
            statusDiv.textContent = '⏸️ Click Start or press S';
            statusDiv.style.color = '#6c757d';
        };
        
        // Cleanup
        window.addEventListener('beforeunload', function() {
            stopCamera();
        });
        
        // Menerima hasil dari Streamlit (untuk update langsung)
        window.addEventListener('message', function(event) {
            const data = event.data;
            if (data && data.type === 'prediction_result') {
                updateResult(data.label, data.confidence);
                statusDiv.textContent = `✅ Predicted: ${data.label}`;
                statusDiv.style.color = '#28a745';
            }
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
        - 📷 Kamera Real-Time
        
        **Keyboard Shortcuts:**
        - `S` = Start Camera
        - `X` = Stop Camera
        - `C` = Capture Abjad
        - `K` = Capture Kata
        
        **Dataset:**
        - Abjad: 26 huruf
        - Kata: 24 kata
        """)
        st.divider()
        st.caption("© 2024 BISINDO Classification")
    
    tab1, tab2, tab3 = st.tabs([
        "📸 Gambar", "🎬 Video", "📷 Kamera Real-Time"
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
    
    # ==================== TAB 3: KAMERA REAL-TIME ====================
    with tab3:
        st.markdown("""
        <div class="card">
            <h3>📷 Kamera Real-Time <span class="live-badge">LIVE</span></h3>
            <p>Deteksi gestur secara real-time dari kamera - hasil langsung muncul di layar</p>
            <p style="color: red; font-size: 0.9rem;">⚠️ Izinkan akses kamera saat diminta browser</p>
            <p style="font-size: 0.9rem;">
                💡 <b>Shortcuts:</b> <kbd>S</kbd> Start | <kbd>X</kbd> Stop | <kbd>C</kbd> Capture Abjad | <kbd>K</kbd> Capture Kata
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # HTML Kamera Real-Time dengan result display langsung
        st.components.v1.html(realtime_camera_html(), height=620)

# ============================================
# LISTENER UNTUK DATA DARI JAVASCRIPT
# ============================================

# Proses data dari query params
query_params = st.query_params

if 'camera_image' in query_params:
    try:
        image_data = query_params['camera_image']
        mode = query_params.get('camera_mode', 'realtime')
        
        # Decode image
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        
        # Load models
        image_model, video_model, image_class_names, video_class_names = load_models()
        
        if image_model is not None:
            if mode == 'kata':
                # Untuk kata, gunakan video model dengan sequence
                # Simpan ke session
                st.session_state.kata_image = img
                label = "Kata: Gunakan video"
                confidence = 0
            elif mode == 'realtime':
                # Real-time prediction (abjad otomatis)
                label, confidence, _ = predict_image(image_model, img, image_class_names)
                st.session_state.camera_label = label
                st.session_state.camera_confidence = confidence
                st.session_state.camera_mode = 'realtime'
                
                # Kirim hasil balik ke JavaScript via URL parameter
                # Hasil akan ditampilkan di JavaScript
            else:
                # Manual capture (abjad)
                label, confidence, _ = predict_image(image_model, img, image_class_names)
                st.session_state.camera_label = label
                st.session_state.camera_confidence = confidence
                st.session_state.camera_mode = 'abjad'
            
            # Clear query params setelah diproses
            st.query_params.clear()
            st.rerun()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
