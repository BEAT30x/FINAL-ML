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
# 3. LOAD CORE MODELS - DENGAN FALLBACK PATH
# ============================================
@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Coba berbagai kemungkinan nama file
        image_names = ["model_gambar.h5", "image_model.h5"]
        video_names = ["video_model.h5"]
        
        image_model = None
        video_model = None
        image_class_names = None
        video_class_names = None
        
        # Cari di root dan folder models
        for img_name in image_names:
            for path in [BASE_DIR, os.path.join(BASE_DIR, "models")]:
                img_path = os.path.join(path, img_name)
                if os.path.exists(img_path):
                    try:
                        image_model = tf.keras.models.load_model(img_path)
                        break
                    except:
                        continue
            if image_model is not None:
                break
        
        # Cari video model
        for vid_name in video_names:
            for path in [BASE_DIR, os.path.join(BASE_DIR, "models")]:
                vid_path = os.path.join(path, vid_name)
                if os.path.exists(vid_path):
                    try:
                        video_model = tf.keras.models.load_model(vid_path)
                        break
                    except:
                        continue
            if video_model is not None:
                break
        
        # Cari class names
        for pkl_name in ["image_class_names.pkl", "video_class_names.pkl"]:
            for path in [BASE_DIR, os.path.join(BASE_DIR, "models")]:
                pkl_path = os.path.join(path, pkl_name)
                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, "rb") as f:
                            if "image" in pkl_name:
                                image_class_names = pickle.load(f)
                            else:
                                video_class_names = pickle.load(f)
                        break
                    except:
                        continue
        
        # Fallback
        if image_class_names is None:
            image_class_names = [chr(65 + i) for i in range(26)]
        if video_class_names is None:
            video_class_names = []
        
        if image_model is None:
            st.error("❌ Model tidak ditemukan!")
            st.code(f"""
            File di root folder:
            {os.listdir(BASE_DIR)}
            """)
            return None, None, None, None
        
        st.success(f"✅ Model berhasil dimuat! {len(image_class_names)} kelas")
        return image_model, video_model, image_class_names, video_class_names
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
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
# [TAMBAHAN] FUNGSI UNTUK BROWSER CAMERA
# ============================================
def browser_camera_html():
    return """
    <div style="text-align: center;">
        <video id="video" width="100%" height="auto" autoplay style="max-height: 400px; background: #000; border-radius: 10px;"></video>
        <br>
        <button id="captureBtn" style="background: #4f46e5; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; margin-top: 10px;">
            📸 Capture & Predict
        </button>
        <canvas id="canvas" style="display: none;"></canvas>
        <div id="status" style="margin-top: 10px; font-weight: bold;">⏸️ Klik Capture</div>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statusDiv = document.getElementById('status');
        const captureBtn = document.getElementById('captureBtn');
        
        let stream = null;
        
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
            .then(mediaStream => {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.play();
                statusDiv.innerHTML = '🟢 Camera ready!';
                statusDiv.style.color = '#28a745';
            })
            .catch(err => {
                statusDiv.innerHTML = '❌ Error: ' + err.message;
                statusDiv.style.color = 'red';
            });
        
        captureBtn.addEventListener('click', function() {
            if (!stream) {
                statusDiv.innerHTML = '⚠️ Kamera belum siap!';
                statusDiv.style.color = 'orange';
                return;
            }
            
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            const currentUrl = window.location.href.split('?')[0];
            window.location.href = currentUrl + '?img=' + encodeURIComponent(imageData) + '&t=' + Date.now();
            
            statusDiv.innerHTML = '⏳ Memproses...';
            statusDiv.style.color = '#667eea';
        });
    </script>
    """

def predict_from_browser_image(img_data):
    try:
        img_b64 = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        processed = preprocess_image(img)
        probs = image_model.predict(processed, verbose=0)[0]
        pred_idx = np.argmax(probs)
        return image_class_names[pred_idx], probs[pred_idx] * 100
    except:
        return None, 0

# ============================================
# 5. SIDEBAR CONTROL PANEL
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
# 6. MAIN CONTROLLER
# ============================================
if image_model is None:
    st.error("🚨 Master Model File tidak ditemukan!")
    st.info("📁 Pastikan file model ada di ROOT folder.")
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
                        if processed_vid is not None and video_model is not None:
                            probs = video_model.predict(processed_vid, verbose=0)[0]
                            pred_idx = np.argmax(probs)
                            
                            label = video_class_names[pred_idx] if video_class_names else f"Kelas {pred_idx}"
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
                                lbl = video_class_names[idx] if video_class_names else f"Kelas {idx}"
                                st.text(f"Kata ({lbl}): {probs[idx]*100:.1f}%")
                                st.progress(float(probs[idx]))
                        else:
                            st.error("Gagal mengurai susunan berkas video.")
            else:
                st.info("💡 Unggah berkas cuplikan rekaman video kata isyarat untuk memulai penafsiran.")

    # ----------------------------------------------------------------
    # TAB 3: LIVE STREAM CAMERA (ABJAD) - BROWSER VERSION
    # ----------------------------------------------------------------
    with tab3:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Abjad</h3><p>Gunakan kamera browser untuk menerjemahkan abjad secara instan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            # PAKAI BROWSER CAMERA BUKAN CV2
            st.components.v1.html(browser_camera_html(), height=500)
            st.info("📷 Klik 'Capture & Predict' untuk mengambil gambar dan prediksi")
        with c2:
            st.markdown("##### 📊 LOG PREDIKSI SISTEM")
            lbl_placeholder = st.empty()
            conf_placeholder = st.empty()
            
            # Proses hasil dari browser capture
            query_params = st.query_params
            if 'img' in query_params:
                try:
                    img_data = query_params['img']
                    label, conf = predict_from_browser_image(img_data)
                    if label:
                        lbl_placeholder.markdown(f"""
                        <div class="kpi-box">
                            <div class="kpi-title">Karakter Terbaca</div>
                            <div class="kpi-value" style="color: #4f46e5;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        conf_placeholder.progress(float(conf / 100), text=f"Tingkat Kepastian Akurasi: {conf:.1f}%")
                    st.query_params.clear()
                except:
                    pass

    # ----------------------------------------------------------------
    # TAB 4: LIVE STREAM CAMERA (KATA)
    # ----------------------------------------------------------------
    with tab4:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Kata (Sistem Sekuensial)</h3><p>Model mengumpulkan 20 runtunan bingkai gambar secara berkesinambungan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            run_kata = st.toggle("Aktifkan Jalur Kamera Kata", key="tg_kata")
            frame_window_kata = st.empty()
        with c2:
            st.markdown("##### 📊 ANALISIS BINGKAI TEMPORAL")
            buffer_progress = st.empty()
            kata_lbl = st.empty()
            
        if run_kata:
            cap = cv2.VideoCapture(0)
            live_buf = deque(maxlen=20)
            
            if not cap.isOpened():
                st.error("Gagal memuat sensor kamera aktif.")
            else:
                while run_kata:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    live_buf.append(frame)
                    
                    current_len = len(live_buf)
                    buffer_progress.progress(float(current_len / 20), text=f"Stabilitas Buffer Isyarat: {current_len}/20 Frames Packed")
                    
                    if current_len == 20 and video_model is not None:
                        label, conf = predict_live_kata(video_model, list(live_buf), video_class_names)
                        kata_lbl.markdown(f"""
                        <div class="kpi-box" style="border-left-color: #10b981;">
                            <div class="kpi-title">Terjemahan Frasa/Kata</div>
                            <div class="kpi-value" style="color: #059669;">"{label.upper()}"</div>
                            <small style="color: #64748b;">Akurasi Sekuensial: {conf:.1f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        kata_lbl.warning("Menyelaraskan data sekuens... Pertahankan posisi tangan Anda di depan kamera.")
                        
                    frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window_kata.image(frame_disp, channels="RGB", use_container_width=True)
                    
                    time.sleep(0.03)
                cap.release()
        else:
            frame_window_kata.info("Aktifkan toggle di atas untuk mulai memproses data gestur berbasis waktu secara interaktif.")
