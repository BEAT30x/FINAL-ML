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
# 3. LOAD CORE MODELS
# FIX: Cari model di root folder DAN folder models/
# ============================================
@st.cache_resource
def load_models():
    try:
        # Semua file ada di ROOT folder (sama dengan app.py)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        image_model       = None
        video_model       = None
        image_class_names = None
        video_class_names = None

        img_path  = os.path.join(BASE_DIR, "image_model.h5")
        vid_path  = os.path.join(BASE_DIR, "video_model.h5")
        ipkl_path = os.path.join(BASE_DIR, "image_class_names.pkl")
        vpkl_path = os.path.join(BASE_DIR, "video_class_names.pkl")

        # --- Cek file ada & ukurannya wajar (bukan pointer Git LFS) ---
        def is_valid_h5(path):
            if not os.path.exists(path):
                return False, "File tidak ditemukan"
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 0.1:
                return False, f"File terlalu kecil ({size_mb:.2f} MB) — kemungkinan Git LFS pointer, bukan file asli"
            return True, f"{size_mb:.1f} MB"

        img_valid, img_info = is_valid_h5(img_path)
        vid_valid, vid_info = is_valid_h5(vid_path)

        # --- Load image model ---
        if img_valid:
            try:
                image_model = tf.keras.models.load_model(img_path)
            except Exception as e:
                st.error(f"❌ Gagal load image_model.h5 ({img_info}): {e}")
        else:
            st.error(f"❌ image_model.h5 — {img_info}")

        # --- Load video model ---
        if vid_valid:
            try:
                video_model = tf.keras.models.load_model(vid_path)
            except Exception as e:
                st.warning(f"⚠️ Gagal load video_model.h5 ({vid_info}): {e}")
        else:
            st.warning(f"⚠️ video_model.h5 — {vid_info}")

        # --- Load class names ---
        try:
            with open(ipkl_path, "rb") as f:
                image_class_names = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Gagal load image_class_names.pkl: {e}")
            image_class_names = [chr(65 + i) for i in range(26)]

        try:
            with open(vpkl_path, "rb") as f:
                video_class_names = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Gagal load video_class_names.pkl: {e}")
            video_class_names = []

        return image_model, video_model, image_class_names, video_class_names

    except Exception as e:
        st.error(f"❌ Error tidak terduga: {e}")
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

def predict_frame_gambar(model, pil_image, class_names):
    """Prediksi dari PIL Image (untuk st.camera_input)"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = pil_image.resize((224, 224))
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
    st.error("🚨 Master Model File (`.h5` / `.pkl`) tidak ditemukan.")
    st.info("📁 Pastikan `image_model.h5` dan file pkl ada di root folder atau folder `models/`.")
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
                    if video_model is None:
                        st.error("❌ Model video tidak ditemukan di repository.")
                    else:
                        with st.spinner("Memproses urutan struktur temporal video..."):
                            processed_vid = preprocess_video(uploaded_vid)
                            if processed_vid is not None:
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
    # TAB 3: LIVE STREAM REAL-TIME (ABJAD) — JavaScript WebRTC
    # Capture otomatis tiap 1.5 detik dari browser, kirim ke Python
    # ----------------------------------------------------------------
    with tab3:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Abjad</h3><p>Kamera berjalan otomatis dan menerjemahkan abjad secara real-time.</p></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1], gap="medium")

        with c1:
            # Ambil frame dari query param jika ada (hasil kiriman JS)
            q = st.query_params
            frame_data_abjad = q.get("frame_abjad", None)

            # Tampilkan komponen kamera JS — auto-capture tiap 1.5 detik
            st.components.v1.html("""
            <div style="font-family:Inter,sans-serif; text-align:center;">
                <video id="vid3" autoplay playsinline muted
                    style="width:100%; max-height:360px; border-radius:12px;
                           background:#000; border:1px solid #cbd5e1;"></video>
                <canvas id="cvs3" style="display:none;"></canvas>
                <div style="margin-top:10px; display:flex; gap:10px; justify-content:center;">
                    <button id="startBtn3"
                        style="background:#4f46e5;color:white;padding:10px 24px;
                               border:none;border-radius:8px;font-size:14px;cursor:pointer;">
                        ▶ Mulai Real-time
                    </button>
                    <button id="stopBtn3"
                        style="background:#ef4444;color:white;padding:10px 24px;
                               border:none;border-radius:8px;font-size:14px;cursor:pointer;display:none;">
                        ■ Hentikan
                    </button>
                </div>
                <div id="status3" style="margin-top:8px;font-size:13px;color:#64748b;">
                    ⏸ Klik Mulai untuk mengaktifkan kamera real-time
                </div>
            </div>
            <script>
                const vid3     = document.getElementById('vid3');
                const cvs3     = document.getElementById('cvs3');
                const ctx3     = cvs3.getContext('2d');
                const startBtn = document.getElementById('startBtn3');
                const stopBtn  = document.getElementById('stopBtn3');
                const status3  = document.getElementById('status3');
                let stream3    = null;
                let intervalId = null;

                startBtn.addEventListener('click', async () => {
                    try {
                        stream3 = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}});
                        vid3.srcObject = stream3;
                        await vid3.play();
                        startBtn.style.display = 'none';
                        stopBtn.style.display  = 'inline-block';
                        status3.innerHTML = '🟢 Kamera aktif — prediksi berjalan otomatis...';
                        status3.style.color = '#166534';

                        // Capture & kirim tiap 1.5 detik
                        intervalId = setInterval(() => {
                            cvs3.width  = vid3.videoWidth  || 640;
                            cvs3.height = vid3.videoHeight || 480;
                            ctx3.drawImage(vid3, 0, 0, cvs3.width, cvs3.height);
                            const b64 = cvs3.toDataURL('image/jpeg', 0.6);
                            const base = window.location.href.split('?')[0];
                            window.location.replace(base + '?frame_abjad=' + encodeURIComponent(b64));
                        }, 1500);

                    } catch(e) {
                        status3.innerHTML = '❌ Error kamera: ' + e.message;
                        status3.style.color = 'red';
                    }
                });

                stopBtn.addEventListener('click', () => {
                    clearInterval(intervalId);
                    if (stream3) stream3.getTracks().forEach(t => t.stop());
                    vid3.srcObject = null;
                    stopBtn.style.display  = 'none';
                    startBtn.style.display = 'inline-block';
                    status3.innerHTML = '⏹ Kamera dihentikan.';
                    status3.style.color = '#64748b';
                });
            </script>
            """, height=460)

        with c2:
            st.markdown("##### 📊 LOG PREDIKSI SISTEM")
            lbl_placeholder  = st.empty()
            conf_placeholder = st.empty()
            hist_placeholder = st.empty()

        # Proses frame yang dikirim JS
        if frame_data_abjad:
            try:
                import base64, io
                b64 = frame_data_abjad.split(',')[1] if ',' in frame_data_abjad else frame_data_abjad
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes))
                label, conf = predict_frame_gambar(image_model, img, image_class_names)

                lbl_placeholder.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-title">Karakter Terbaca</div>
                    <div class="kpi-value" style="color:#4f46e5;">{label}</div>
                </div>""", unsafe_allow_html=True)
                conf_placeholder.progress(float(conf / 100), text=f"Tingkat Kepastian Akurasi: {conf:.1f}%")

                # Simpan histori prediksi
                if "abjad_history" not in st.session_state:
                    st.session_state.abjad_history = []
                st.session_state.abjad_history.append(label)
                if len(st.session_state.abjad_history) > 10:
                    st.session_state.abjad_history.pop(0)

                hist_placeholder.markdown(
                    "**Histori:** " + " → ".join(st.session_state.abjad_history)
                )
                st.query_params.clear()
            except Exception as e:
                lbl_placeholder.error(f"Error prediksi: {e}")
        else:
            lbl_placeholder.info("Klik **▶ Mulai Real-time** untuk mengaktifkan kamera.")

    # ----------------------------------------------------------------
    # TAB 4: LIVE STREAM REAL-TIME (KATA) — JS WebRTC + buffer otomatis
    # Capture otomatis tiap 0.5 detik, kumpulkan 20 frame, prediksi kata
    # ----------------------------------------------------------------
    with tab4:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Kata (Sistem Sekuensial)</h3><p>Kamera mengumpulkan 20 frame otomatis secara berkesinambungan untuk prediksi kata.</p></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1], gap="medium")

        with c1:
            q = st.query_params
            frame_data_kata = q.get("frame_kata", None)

            st.components.v1.html("""
            <div style="font-family:Inter,sans-serif; text-align:center;">
                <video id="vid4" autoplay playsinline muted
                    style="width:100%; max-height:360px; border-radius:12px;
                           background:#000; border:1px solid #cbd5e1;"></video>
                <canvas id="cvs4" style="display:none;"></canvas>
                <div style="margin-top:10px; display:flex; gap:10px; justify-content:center;">
                    <button id="startBtn4"
                        style="background:#10b981;color:white;padding:10px 24px;
                               border:none;border-radius:8px;font-size:14px;cursor:pointer;">
                        ▶ Mulai Rekam Kata
                    </button>
                    <button id="stopBtn4"
                        style="background:#ef4444;color:white;padding:10px 24px;
                               border:none;border-radius:8px;font-size:14px;cursor:pointer;display:none;">
                        ■ Hentikan
                    </button>
                </div>
                <div id="status4" style="margin-top:8px;font-size:13px;color:#64748b;">
                    ⏸ Klik Mulai untuk mengumpulkan frame kata secara otomatis
                </div>
            </div>
            <script>
                const vid4      = document.getElementById('vid4');
                const cvs4      = document.getElementById('cvs4');
                const ctx4      = cvs4.getContext('2d');
                const startBtn4 = document.getElementById('startBtn4');
                const stopBtn4  = document.getElementById('stopBtn4');
                const status4   = document.getElementById('status4');
                let stream4     = null;
                let interval4   = null;
                let frameCount  = 0;

                startBtn4.addEventListener('click', async () => {
                    try {
                        stream4 = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}});
                        vid4.srcObject = stream4;
                        await vid4.play();
                        startBtn4.style.display = 'none';
                        stopBtn4.style.display  = 'inline-block';
                        frameCount = 0;

                        interval4 = setInterval(() => {
                            frameCount++;
                            status4.innerHTML = '🟢 Mengumpulkan frame ' + frameCount + '/20...';
                            status4.style.color = '#166534';

                            cvs4.width  = vid4.videoWidth  || 640;
                            cvs4.height = vid4.videoHeight || 480;
                            ctx4.drawImage(vid4, 0, 0, cvs4.width, cvs4.height);
                            const b64  = cvs4.toDataURL('image/jpeg', 0.5);
                            const base = window.location.href.split('?')[0];
                            window.location.replace(base + '?frame_kata=' + encodeURIComponent(b64));

                            if (frameCount >= 20) {
                                frameCount = 0;
                                status4.innerHTML = '🔄 Prediksi dijalankan, mengulang buffer...';
                            }
                        }, 600);

                    } catch(e) {
                        status4.innerHTML = '❌ Error kamera: ' + e.message;
                        status4.style.color = 'red';
                    }
                });

                stopBtn4.addEventListener('click', () => {
                    clearInterval(interval4);
                    if (stream4) stream4.getTracks().forEach(t => t.stop());
                    vid4.srcObject = null;
                    stopBtn4.style.display  = 'none';
                    startBtn4.style.display = 'inline-block';
                    status4.innerHTML = '⏹ Kamera dihentikan.';
                    status4.style.color = '#64748b';
                });
            </script>
            """, height=460)

        with c2:
            st.markdown("##### 📊 ANALISIS BINGKAI TEMPORAL")
            buffer_progress = st.empty()
            kata_lbl        = st.empty()

        # Inisialisasi buffer di session_state
        if "kata_frame_buffer" not in st.session_state:
            st.session_state.kata_frame_buffer = []

        # Proses frame yang dikirim JS
        if frame_data_kata:
            try:
                import base64, io
                b64 = frame_data_kata.split(',')[1] if ',' in frame_data_kata else frame_data_kata
                img_bytes = base64.b64decode(b64)
                img_kata  = Image.open(io.BytesIO(img_bytes)).resize((96, 96))
                frame_arr = np.array(img_kata, dtype=np.float32) / 255.0

                st.session_state.kata_frame_buffer.append(frame_arr)
                if len(st.session_state.kata_frame_buffer) > 20:
                    st.session_state.kata_frame_buffer.pop(0)

                current_len = len(st.session_state.kata_frame_buffer)
                buffer_progress.progress(
                    float(current_len / 20),
                    text=f"Stabilitas Buffer Isyarat: {current_len}/20 Frames Packed"
                )

                if current_len == 20 and video_model is not None and video_class_names:
                    frames_np   = np.array(st.session_state.kata_frame_buffer, dtype=np.float32)
                    video_arr   = np.expand_dims(frames_np, axis=0)
                    predictions = video_model.predict(video_arr, verbose=0)
                    pred_class  = np.argmax(predictions[0])
                    label       = video_class_names[pred_class]
                    conf        = float(predictions[0][pred_class] * 100)

                    kata_lbl.markdown(f"""
                    <div class="kpi-box" style="border-left-color: #10b981;">
                        <div class="kpi-title">Terjemahan Frasa/Kata</div>
                        <div class="kpi-value" style="color: #059669;">"{label.upper()}"</div>
                        <small style="color: #64748b;">Akurasi Sekuensial: {conf:.1f}%</small>
                    </div>""", unsafe_allow_html=True)

                    # Reset buffer setelah prediksi agar siklus baru dimulai
                    st.session_state.kata_frame_buffer = []
                elif video_model is None:
                    kata_lbl.error("❌ Model video tidak tersedia.")
                else:
                    kata_lbl.warning("Menyelaraskan data sekuens... Pertahankan posisi tangan Anda di depan kamera.")

                st.query_params.clear()
            except Exception as e:
                kata_lbl.error(f"Error buffer frame: {e}")
        else:
            kata_lbl.info("Klik **▶ Mulai Rekam Kata** untuk memulai pengumpulan frame otomatis.")
