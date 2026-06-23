import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import tempfile
import time
from collections import deque
import base64
import io

# ============================================
# 1. KONFIGURASI HALAMAN & TEMA UTAMA
# ============================================
st.set_page_config(
    page_title="BISINDO Intelligence Translator",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema UI/UX Profesional (Clean Minimalist & Enterprise Look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    .dashboard-header {
        background: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title-section h1 {
        font-size: 1.8rem;
        color: #0f172a;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-title-section p {
        font-size: 0.95rem;
        color: #64748b;
        margin: 0.25rem 0 0 0;
    }
    .badge-status {
        background-color: #f0fdf4;
        color: #166534;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #bbf7d0;
    }
    
    .pro-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .pro-card h3 {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .pro-card p {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1rem;
    }

    .kpi-box {
        background: #fdfdfd;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.25rem 0;
    }
    
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        color: #64748b;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4f46e5 !important;
        font-weight: 600;
    }
    
    .video-stream-container img {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    .camera-container {
        border: 3px solid #4f46e5;
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
# 2. APPLICATION HEADER
# ============================================
st.markdown("""
<div class="dashboard-header">
    <div class="header-title-section">
        <h1>🖐️ BISINDO Intelligence System</h1>
        <p>Platform enterprise klasifikasi dan penerjemah Bahasa Isyarat Indonesia secara real-time</p>
    </div>
    <div>
        <span class="badge-status">● Engine V2.0 Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# 3. LOAD CORE MODELS
# ============================================
@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        image_model_path = os.path.join(BASE_DIR, "image_model.h5")
        video_model_path = os.path.join(BASE_DIR, "video_model.h5")
        image_pkl_path = os.path.join(BASE_DIR, "image_class_names.pkl")
        video_pkl_path = os.path.join(BASE_DIR, "video_class_names.pkl")
        
        if not os.path.exists(image_model_path):
            image_model_path = os.path.join(BASE_DIR, "models", "image_model.h5")
            video_model_path = os.path.join(BASE_DIR, "models", "video_model.h5")
            image_pkl_path = os.path.join(BASE_DIR, "models", "image_class_names.pkl")
            video_pkl_path = os.path.join(BASE_DIR, "models", "video_class_names.pkl")
        
        image_model = tf.keras.models.load_model(image_model_path)
        video_model = tf.keras.models.load_model(video_model_path)
        
        with open(image_pkl_path, "rb") as f:
            image_class_names = pickle.load(f)
        with open(video_pkl_path, "rb") as f:
            video_class_names = pickle.load(f)
        
        return image_model, video_model, image_class_names, video_class_names
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None

image_model, video_model, image_class_names, video_class_names = load_models()

# ============================================
# 4. PREPROCESSING PIPELINES
# ============================================
def preprocess_image(image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_video(video_file, max_frames=20, img_size=96):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    try: os.remove(tmp_path)
    except: pass
        
    total_frames = len(all_frames)
    if total_frames == 0: return None
    
    frames = []
    indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    for idx in indices:
        if idx < total_frames:
            frame = all_frames[idx]
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
    
    return np.expand_dims(np.array(frames, dtype=np.float32), axis=0)

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

def predict_live_kata(model, frame_list, class_names, img_size=96):
    frames = []
    for frame in frame_list:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        frames.append(frame_resized.astype(np.float32) / 255.0)
    
    video_array = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)
    predictions = model.predict(video_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    return class_names[pred_class], predictions[0][pred_class] * 100

# ============================================
# 5. FUNGSI PREDIKSI DARI BYTES
# ============================================
def predict_image_from_bytes(image_bytes, model, class_names):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    return class_names[pred_class], predictions[0][pred_class] * 100

# ============================================
# 6. HTML KAMERA BROWSER (ABJAD)
# ============================================
def camera_abjad_html():
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-container">
            <video id="videoAbjad" width="100%" height="auto" autoplay style="max-height: 400px; background: #000;"></video>
            <div id="overlayAbjad" class="overlay-label" style="color: #ffcc00;">🔄 Menunggu gestur...</div>
        </div>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button id="startAbjad" style="background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ▶️ Start Camera
            </button>
            <button id="stopAbjad" style="background: #dc3545; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
            <button id="captureAbjad" style="background: #4f46e5; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                📸 Capture
            </button>
        </div>
        <br>
        <div id="statusAbjad" style="font-size: 14px; font-weight: bold; color: #6c757d; min-height: 25px;">
            ⏸️ Klik Start untuk mengaktifkan kamera
        </div>
        <canvas id="canvasAbjad" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('videoAbjad');
        const canvas = document.getElementById('canvasAbjad');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('overlayAbjad');
        const statusDiv = document.getElementById('statusAbjad');
        const startBtn = document.getElementById('startAbjad');
        const stopBtn = document.getElementById('stopAbjad');
        const captureBtn = document.getElementById('captureAbjad');
        
        let stream = null;
        let isRunning = false;
        
        function sendToStreamlit(imageData) {
            const currentUrl = window.location.href.split('?')[0];
            window.location.href = currentUrl + '?img=' + encodeURIComponent(imageData) + '&mode=abjad&t=' + Date.now();
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
            overlay.textContent = '⏳ Memproses...';
            overlay.style.color = '#ffcc00';
            sendToStreamlit(imageData);
        }
        
        function startCamera() {
            if (isRunning) return;
            
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } })
            .then(mediaStream => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.play();
                isRunning = true;
                statusDiv.textContent = '🟢 Camera running...';
                statusDiv.style.color = '#28a745';
                overlay.textContent = '🔄 Siap capture!';
                overlay.style.color = '#00ff00';
                startBtn.disabled = true;
                startBtn.style.opacity = '0.6';
                stopBtn.disabled = false;
            })
            .catch(err => {
                statusDiv.textContent = '❌ Error: ' + err.message;
                statusDiv.style.color = 'red';
                overlay.textContent = '❌ Camera Error';
                overlay.style.color = 'red';
            });
        }
        
        function stopCamera() {
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
            overlay.textContent = '⏸️ Stopped';
            overlay.style.color = '#ffffff';
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
            statusDiv.textContent = '⏸️ Klik Start atau tekan S';
        };
        
        window.addEventListener('beforeunload', function() {
            stopCamera();
        });
    </script>
    """

# ============================================
# 7. HTML KAMERA BROWSER (KATA)
# ============================================
def camera_kata_html():
    return """
    <div style="text-align: center; margin: 10px 0;">
        <div class="camera-container">
            <video id="videoKata" width="100%" height="auto" autoplay style="max-height: 400px; background: #000;"></video>
            <div id="overlayKata" class="overlay-label" style="color: #ffcc00;">🔄 Menunggu gestur...</div>
        </div>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button id="startKata" style="background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ▶️ Start Camera
            </button>
            <button id="stopKata" style="background: #dc3545; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                ⏹️ Stop
            </button>
            <button id="captureKata" style="background: #10b981; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; font-weight: bold;">
                🎬 Capture (Kata)
            </button>
        </div>
        <br>
        <div id="statusKata" style="font-size: 14px; font-weight: bold; color: #6c757d; min-height: 25px;">
            ⏸️ Klik Start untuk mengaktifkan kamera
        </div>
        <div id="bufferInfo" style="font-size: 13px; color: #64748b; margin-top: 5px;">
            Buffer: 0/20 frame
        </div>
        <canvas id="canvasKata" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('videoKata');
        const canvas = document.getElementById('canvasKata');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('overlayKata');
        const statusDiv = document.getElementById('statusKata');
        const bufferInfo = document.getElementById('bufferInfo');
        const startBtn = document.getElementById('startKata');
        const stopBtn = document.getElementById('stopKata');
        const captureBtn = document.getElementById('captureKata');
        
        let stream = null;
        let isRunning = false;
        let buffer = [];
        
        function sendToStreamlit(imageData) {
            const currentUrl = window.location.href.split('?')[0];
            window.location.href = currentUrl + '?img=' + encodeURIComponent(imageData) + '&mode=kata&t=' + Date.now();
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
            overlay.textContent = '⏳ Memproses...';
            overlay.style.color = '#ffcc00';
            sendToStreamlit(imageData);
        }
        
        function startCamera() {
            if (isRunning) return;
            
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } })
            .then(mediaStream => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.play();
                isRunning = true;
                buffer = [];
                statusDiv.textContent = '🟢 Camera running...';
                statusDiv.style.color = '#28a745';
                overlay.textContent = '🔄 Siap capture!';
                overlay.style.color = '#00ff00';
                bufferInfo.textContent = 'Buffer: 0/20 frame';
                startBtn.disabled = true;
                startBtn.style.opacity = '0.6';
                stopBtn.disabled = false;
            })
            .catch(err => {
                statusDiv.textContent = '❌ Error: ' + err.message;
                statusDiv.style.color = 'red';
                overlay.textContent = '❌ Camera Error';
                overlay.style.color = 'red';
            });
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            if (video.srcObject) {
                video.srcObject = null;
            }
            isRunning = false;
            buffer = [];
            startBtn.disabled = false;
            startBtn.style.opacity = '1';
            statusDiv.textContent = '⏸️ Stopped';
            statusDiv.style.color = '#6c757d';
            overlay.textContent = '⏸️ Stopped';
            overlay.style.color = '#ffffff';
            bufferInfo.textContent = 'Buffer: 0/20 frame';
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
            statusDiv.textContent = '⏸️ Klik Start atau tekan S';
        };
        
        window.addEventListener('beforeunload', function() {
            stopCamera();
        });
    </script>
    """

# ============================================
# 8. SIDEBAR CONTROL PANEL
# ============================================
with st.sidebar:
    st.markdown("### 🛠️ System Control & Core Info")
    st.info("Aplikasi berjalan dalam mode inferensi performa tinggi memanfaatkan optimasi MobileNetV2.")
    st.markdown("---")
    st.markdown("**Struktur Kelas Terdaftar:**")
    st.write(f"• **Model Statis:** {len(image_class_names) if image_class_names else 0} Alfabet Terpetakan")
    st.write(f"• **Model Dinamis:** {len(video_class_names) if video_class_names else 0} Kata Terpetakan")
    st.markdown("---")
    st.caption("© 2026 BISINDO Neural Translator Network Pro Tier")

# ============================================
# 9. MAIN CONTROLLER
# ============================================
if image_model is None or video_model is None:
    st.error("🚨 Master Model File (`.h5` / `.pkl`) tidak ditemukan!")
    st.info("📁 Pastikan file model ada di ROOT folder atau folder `models/`:")
    st.code("""
    📁 BISINDO_Deployment/
    ├── app.py
    ├── image_model.h5       ← Harus ada!
    ├── video_model.h5       ← Harus ada!
    ├── image_class_names.pkl
    ├── video_class_names.pkl
    └── models/              (opsional)
        ├── image_model.h5
        ├── video_model.h5
        ├── image_class_names.pkl
        └── video_class_names.pkl
    """)
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Unggah Citra (Abjad)",
        "📂 Unggah Video (Kata)",
        "⚡ Live Stream (Abjad/Huruf)",
        "⚡ Live Stream (Kata/Kalimat)"
    ])

    # ----------------------------------------------------------------
    # TAB 1: ASSET IMAGE UPLOADER
    # ----------------------------------------------------------------
    with tab1:
        st.markdown('<div class="pro-card"><h3>Analisis File Gambar Statis</h3><p>Sistem akan memindai kontur geometris tangan dari berkas gambar yang diunggah.</p></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            uploaded_img = st.file_uploader("Unggah berkas gambar (JPG, PNG)", type=["jpg", "jpeg", "png"], key="img_pro")
            if uploaded_img:
                image = Image.open(uploaded_img)
                st.image(image, caption="Preview Asset", use_container_width=True)
        
        with col2:
            if uploaded_img:
                if st.button("Jalankan Inferensi Gambar", type="primary", use_container_width=True):
                    with st.spinner("Menghitung matriks probabilitas..."):
                        processed = preprocess_image(image)
                        probs = image_model.predict(processed, verbose=0)[0]
                        pred_idx = np.argmax(probs)
                        
                        label = image_class_names[pred_idx]
                        confidence = probs[pred_idx] * 100
                        
                        st.markdown(f"""
                        <div class="kpi-box">
                            <div class="kpi-title">Hasil Klasifikasi Dominan</div>
                            <div class="kpi-value">Abjad: {label}</div>
                            <div style="color: #166534; font-size: 0.9rem; font-weight: 500;">Tingkat Kepercayaan: {confidence:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br><b>Distribusi Keyakinan Model:</b>", unsafe_allow_html=True)
                        top5 = np.argsort(probs)[-5:][::-1]
                        for idx in top5:
                            st.text(f"Kelas [{image_class_names[idx]}]: {probs[idx]*100:.1f}%")
                            st.progress(float(probs[idx]))
            else:
                st.info("💡 Hubungkan dokumen gambar di panel kiri untuk mengaktifkan kalkulasi model.")

    # ----------------------------------------------------------------
    # TAB 2: ASSET VIDEO UPLOADER
    # ----------------------------------------------------------------
    with tab2:
        st.markdown('<div class="pro-card"><h3>Analisis Rekaman Video Isyarat</h3><p>Ekstraksi otomatis untuk mengidentifikasi klasifikasi kata dinamis.</p></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            uploaded_vid = st.file_uploader("Unggah rekaman video (MP4, MOV)", type=["mp4", "mov"], key="vid_pro")
            if uploaded_vid:
                st.video(uploaded_vid)
        
        with col2:
            if uploaded_vid:
                if st.button("Jalankan Inferensi Video", type="primary", use_container_width=True):
                    with st.spinner("Memproses urutan struktur temporal video..."):
                        processed_vid = preprocess_video(uploaded_vid)
                        if processed_vid is not None:
                            probs = video_model.predict(processed_vid, verbose=0)[0]
                            pred_idx = np.argmax(probs)
                            
                            label = video_class_names[pred_idx]
                            confidence = probs[pred_idx] * 100
                            
                            st.markdown(f"""
                            <div class="kpi-box">
                                <div class="kpi-title">Kata Isyarat Terjemahan</div>
                                <div class="kpi-value">"{label.upper()}"</div>
                                <div style="color: #166534; font-size: 0.9rem; font-weight: 500;">Tingkat Konfidensi Sekuensial: {confidence:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<br><b>Top Estimasi Alternatif:</b>", unsafe_allow_html=True)
                            top5_vid = np.argsort(probs)[-5:][::-1]
                            for idx in top5_vid:
                                st.text(f"Kata ({video_class_names[idx]}): {probs[idx]*100:.1f}%")
                                st.progress(float(probs[idx]))
                        else:
                            st.error("Gagal mengurai susunan berkas video.")
            else:
                st.info("💡 Unggah berkas cuplikan rekaman video kata isyarat untuk memulai penafsiran.")

    # ----------------------------------------------------------------
    # TAB 3: LIVE STREAM CAMERA (ABJAD) - VIA BROWSER
    # ----------------------------------------------------------------
    with tab3:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Abjad</h3><p>Gunakan kamera browser untuk menerjemahkan abjad secara instan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            st.components.v1.html(camera_abjad_html(), height=500)
        with c2:
            st.markdown("##### 📊 LOG PREDIKSI SISTEM")
            lbl_placeholder = st.empty()
            conf_placeholder = st.empty()
            
            # Tambahkan info
            st.markdown("""
            <div style="background: #f0fdf4; padding: 1rem; border-radius: 10px; border: 1px solid #bbf7d0; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.85rem;">
                    💡 <b>Tips:</b><br>
                    • Klik <b>Start Camera</b> atau tekan <kbd>S</kbd><br>
                    • Tunjukkan gestur di depan kamera<br>
                    • Klik <b>Capture</b> atau tekan <kbd>C</kbd><br>
                    • Hasil akan muncul di samping
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # TAB 4: LIVE STREAM CAMERA (KATA) - VIA BROWSER
    # ----------------------------------------------------------------
    with tab4:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Kata (Sistem Sekuensial)</h3><p>Model mengumpulkan 20 runtunan bingkai gambar secara berkesinambungan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            st.components.v1.html(camera_kata_html(), height=500)
        with c2:
            st.markdown("##### 📊 ANALISIS BINGKAI TEMPORAL")
            kata_lbl = st.empty()
            
            st.markdown("""
            <div style="background: #f0fdf4; padding: 1rem; border-radius: 10px; border: 1px solid #bbf7d0; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.85rem;">
                    💡 <b>Tips:</b><br>
                    • Klik <b>Start Camera</b> atau tekan <kbd>S</kbd><br>
                    • Tunjukkan gestur kata di depan kamera<br>
                    • Klik <b>Capture (Kata)</b> atau tekan <kbd>C</kbd><br>
                    • Hasil akan muncul di samping
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# 10. LISTENER UNTUK DATA DARI JAVASCRIPT
# ============================================

# Proses data dari query params
query_params = st.query_params

if 'img' in query_params:
    try:
        img_data = query_params['img']
        mode = query_params.get('mode', 'abjad')
        
        if '&t=' in img_data:
            img_data = img_data.split('&t=')[0]
        
        img_b64 = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        
        if mode == 'abjad' and image_model is not None:
            label, confidence = predict_image_from_bytes(img_bytes, image_model, image_class_names)
            
            # Kirim hasil ke JavaScript melalui sesi state
            st.session_state['abjad_result'] = label
            st.session_state['abjad_confidence'] = confidence
            
            st.success(f"🎯 Hasil: {label} ({confidence:.1f}%)")
            
            # Tampilkan di placeholder
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Karakter Terbaca</div>
                <div class="kpi-value" style="color: #4f46e5;">{label}</div>
                <div style="color: #166534; font-size: 0.9rem; font-weight: 500;">Tingkat Kepastian Akurasi: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        elif mode == 'kata' and video_model is not None:
            # Untuk kata, kita butuh beberapa frame
            # Simpan ke buffer session
            if 'kata_buffer' not in st.session_state:
                st.session_state.kata_buffer = []
            
            # Tambahkan frame ke buffer
            st.session_state.kata_buffer.append(img_bytes)
            if len(st.session_state.kata_buffer) > 20:
                st.session_state.kata_buffer = st.session_state.kata_buffer[-20:]
            
            buffer_len = len(st.session_state.kata_buffer)
            
            if buffer_len >= 20:
                # Proses semua frame
                frames = []
                for b in st.session_state.kata_buffer:
                    img = Image.open(io.BytesIO(b))
                    img = img.resize((96, 96))
                    frames.append(np.array(img, dtype=np.float32) / 255.0)
                
                video_array = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)
                probs = video_model.predict(video_array, verbose=0)[0]
                pred_idx = np.argmax(probs)
                
                label = video_class_names[pred_idx]
                confidence = probs[pred_idx] * 100
                
                st.session_state['kata_result'] = label
                st.session_state['kata_confidence'] = confidence
                
                st.success(f"🎯 Hasil Kata: {label} ({confidence:.1f}%)")
                
                st.markdown(f"""
                <div class="kpi-box" style="border-left-color: #10b981;">
                    <div class="kpi-title">Terjemahan Frasa/Kata</div>
                    <div class="kpi-value" style="color: #059669;">"{label.upper()}"</div>
                    <small style="color: #64748b;">Akurasi Sekuensial: {confidence:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Reset buffer setelah prediksi
                st.session_state.kata_buffer = []
            else:
                st.info(f"⏳ Mengumpulkan frame... {buffer_len}/20")
        
        # Clear query params
        st.query_params.clear()
        st.rerun()
        
    except Exception as e:
        print(f"Error: {e}")
        st.query_params.clear()
