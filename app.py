"""
====================================================================
APLIKASI KLASIFIKASI GESTUR TANGAN BISINDO
Bahasa Isyarat Indonesia (BISINDO) - REAL-TIME CAMERA
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
import threading

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
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
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
    """Prediksi dari array numpy"""
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
# REAL-TIME CAMERA HTML
# ============================================

def realtime_camera_html():
    """HTML/JavaScript untuk real-time camera streaming"""
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-preview">
            <video id="video" width="100%" height="auto" autoplay style="max-height: 400px;"></video>
        </div>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button id="startBtn" style="background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ▶️ Start
            </button>
            <button id="stopBtn" style="background: #dc3545; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
            <button id="captureBtn" style="background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                📸 Capture Now
            </button>
        </div>
        <br>
        <div id="status" style="font-size: 16px; font-weight: bold; min-height: 30px;">⏸️ Camera stopped</div>
        <div id="result" style="font-size: 20px; font-weight: bold; min-height: 50px; margin-top: 10px;"></div>
        <canvas id="canvas" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statusDiv = document.getElementById('status');
        const resultDiv = document.getElementById('result');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const captureBtn = document.getElementById('captureBtn');
        
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        let isCapturing = false;
        
        // Fungsi update status
        function updateStatus(text, color) {
            statusDiv.textContent = text;
            statusDiv.style.color = color || '#333';
        }
        
        // Fungsi update result
        function updateResult(text, color) {
            resultDiv.textContent = text;
            resultDiv.style.color = color || '#333';
        }
        
        // Kirim data ke Streamlit
        function sendToStreamlit(imageData, mode) {
            const data = {
                type: 'camera_frame',
                mode: mode,
                image: imageData
            };
            window.parent.postMessage(data, '*');
        }
        
        // Capture frame
        function captureFrame(mode) {
            if (!isRunning) {
                updateResult('⚠️ Start camera dulu!', 'orange');
                return;
            }
            
            isCapturing = true;
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            updateResult('⏳ Memproses...', '#667eea');
            
            // Kirim ke Streamlit dengan mode
            sendToStreamlit(imageData, mode);
            
            setTimeout(() => {
                isCapturing = false;
            }, 100);
        }
        
        // Start camera
        function startCamera() {
            if (isRunning) {
                updateResult('⚠️ Camera sudah berjalan!', 'orange');
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
                updateStatus('🟢 Camera running', '#28a745');
                updateResult('🟢 Camera ready!', '#28a745');
                
                // Mulai real-time prediction setiap 500ms
                if (intervalId) clearInterval(intervalId);
                intervalId = setInterval(() => {
                    if (isRunning && !isCapturing) {
                        // Auto capture untuk real-time
                        canvas.width = video.videoWidth || 640;
                        canvas.height = video.videoHeight || 480;
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const imageData = canvas.toDataURL('image/jpeg', 0.8);
                        sendToStreamlit(imageData, 'realtime');
                    }
                }, 500); // Prediksi setiap 500ms
            })
            .catch(err => {
                updateStatus('❌ Error: ' + err.message, '#dc3545');
                updateResult('❌ Gagal akses kamera!', '#dc3545');
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
            isCapturing = false;
            updateStatus('⏸️ Camera stopped', '#6c757d');
            updateResult('⏸️ Camera stopped', '#6c757d');
        }
        
        // Event listeners
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        captureBtn.addEventListener('click', function() {
            captureFrame('manual');
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 's' || e.key === 'S') {
                startCamera();
            } else if (e.key === 'x' || e.key === 'X') {
                stopCamera();
            } else if (e.key === 'c' || e.key === 'C') {
                captureFrame('manual');
            }
        });
        
        // Notify Streamlit that JavaScript is ready
        window.onload = function() {
            window.parent.postMessage({ type: 'camera_ready' }, '*');
            updateStatus('⏸️ Click Start or press S', '#6c757d');
        };
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            stopCamera();
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
        - `C` = Capture
        
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
            <h3>📷 Kamera Real-Time</h3>
            <p>Deteksi gestur secara real-time dari kamera</p>
            <p style="color: red; font-size: 0.9rem;">⚠️ Izinkan akses kamera saat diminta browser</p>
            <p style="font-size: 0.9rem;">
                💡 <b>Shortcuts:</b> <kbd>S</kbd> Start | <kbd>X</kbd> Stop | <kbd>C</kbd> Capture
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_cam, col_result = st.columns([2, 1])
        
        with col_cam:
            # HTML Kamera Real-Time
            st.components.v1.html(realtime_camera_html(), height=600)
        
        with col_result:
            st.markdown("### 🎯 Hasil Deteksi")
            result_container = st.empty()
            confidence_container = st.empty()
            
            # State untuk hasil kamera
            if 'camera_result' not in st.session_state:
                st.session_state.camera_result = "Menunggu..."
                st.session_state.camera_confidence = 0
                st.session_state.camera_mode = "abjad"
            
            # Tampilkan hasil real-time
            if st.session_state.camera_result:
                if st.session_state.camera_mode == "abjad":
                    result_container.markdown(f"""
                    <div class="result-box fade-in">
                        <h2 style="font-size:2rem;">🎯 {st.session_state.camera_result}</h2>
                        <p class="confidence">Confidence: {st.session_state.camera_confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    result_container.markdown(f"""
                    <div class="result-box-video fade-in">
                        <h2 style="font-size:2rem;">🎯 {st.session_state.camera_result}</h2>
                        <p class="confidence">Confidence: {st.session_state.camera_confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress bar confidence
                confidence_container.progress(st.session_state.camera_confidence / 100)
            
            # Tombol reset
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.camera_result = "Menunggu..."
                st.session_state.camera_confidence = 0
                st.rerun()
            
            # Info
            st.markdown("---")
            st.markdown("""
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 10px; font-size: 0.9rem;">
                <b>📝 Catatan:</b>
                <br>• Kamera akan memprediksi setiap 500ms
                <br>• Gunakan tombol Capture untuk hasil terbaik
                <br>• Pastikan pencahayaan cukup
            </div>
            """, unsafe_allow_html=True)

# ============================================
# LISTENER UNTUK DATA DARI JAVASCRIPT
# ============================================

# Komponen tersembunyi untuk menerima data dari JavaScript
def camera_listener():
    """Menerima data dari JavaScript via postMessage"""
    import streamlit.components.v1 as components
    
    listener_html = """
    <script>
        window.addEventListener('message', function(event) {
            const data = event.data;
            if (data && data.type === 'camera_frame') {
                // Simpan data ke session storage
                sessionStorage.setItem('camera_image', data.image);
                sessionStorage.setItem('camera_mode', data.mode);
                // Kirim ke Streamlit melalui URL parameter
                window.location.href = window.location.href.split('?')[0] + 
                    '?camera_image=' + encodeURIComponent(data.image) +
                    '&camera_mode=' + data.mode;
            }
        });
    </script>
    """
    components.html(listener_html, height=0)

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    # Proses data dari URL jika ada
    import urllib.parse
    
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
                # Prediksi berdasarkan mode
                if mode == 'kata':
                    # Untuk kata, kita perlu buat sequence dari gambar
                    # Simpan gambar ke session untuk diproses oleh video model
                    st.session_state.camera_image = img
                    st.session_state.camera_mode = 'kata'
                    label = "Gunakan fitur video"
                    confidence = 0
                else:
                    label, confidence, _ = predict_image(image_model, img, image_class_names)
                    st.session_state.camera_result = label
                    st.session_state.camera_confidence = confidence
                    st.session_state.camera_mode = 'abjad'
                
                st.rerun()
                
        except Exception as e:
            print(f"Error processing camera data: {e}")
    
    main()
