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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        search_paths = [BASE_DIR, os.path.join(BASE_DIR, "models")]

        image_model      = None
        video_model      = None
        image_class_names = None
        video_class_names = None

        # Load image model
        for fname in ["image_model.h5", "model_gambar.h5"]:
            if image_model is not None:
                break
            for path in search_paths:
                full = os.path.join(path, fname)
                if os.path.exists(full):
                    try:
                        image_model = tf.keras.models.load_model(full)
                        break
                    except:
                        continue

        # Load video model
        for fname in ["video_model.h5"]:
            if video_model is not None:
                break
            for path in search_paths:
                full = os.path.join(path, fname)
                if os.path.exists(full):
                    try:
                        video_model = tf.keras.models.load_model(full)
                        break
                    except:
                        continue

        # Load class names
        for fname in ["image_class_names.pkl", "video_class_names.pkl"]:
            for path in search_paths:
                full = os.path.join(path, fname)
                if os.path.exists(full):
                    try:
                        with open(full, "rb") as f:
                            if "image" in fname:
                                image_class_names = pickle.load(f)
                            else:
                                video_class_names = pickle.load(f)
                        break
                    except:
                        continue

        if image_class_names is None:
            image_class_names = [chr(65 + i) for i in range(26)]
        if video_class_names is None:
            video_class_names = []

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
    # TAB 3: LIVE STREAM CAMERA (ABJAD)
    # FIX: Ganti cv2.VideoCapture(0) → st.camera_input() agar bisa jalan di cloud
    # ----------------------------------------------------------------
    with tab3:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Abjad</h3><p>Gunakan kamera untuk menerjemahkan abjad secara instan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            # st.camera_input bekerja di browser — tidak butuh webcam server
            cam_photo = st.camera_input("📷 Arahkan tangan ke kamera, lalu klik tombol capture", key="cam_abjad")
        with c2:
            st.markdown("##### 📊 LOG PREDIKSI SISTEM")
            lbl_placeholder  = st.empty()
            conf_placeholder = st.empty()
            
        if cam_photo is not None:
            img = Image.open(cam_photo)
            label, conf = predict_frame_gambar(image_model, img, image_class_names)
            
            lbl_placeholder.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Karakter Terbaca</div>
                <div class="kpi-value" style="color: #4f46e5;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            conf_placeholder.progress(float(conf / 100), text=f"Tingkat Kepastian Akurasi: {conf:.1f}%")
        else:
            lbl_placeholder.info("Toggle sakelar di atas untuk mengaktifkan modul jepretan kamera real-time.")

    # ----------------------------------------------------------------
    # TAB 4: LIVE STREAM CAMERA (KATA)
    # FIX: Kumpulkan frame dari st.camera_input berulang kali (buffer manual)
    #      agar kompatibel dengan Streamlit Cloud tanpa cv2.VideoCapture(0)
    # ----------------------------------------------------------------
    with tab4:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Kata (Sistem Sekuensial)</h3><p>Model mengumpulkan 20 runtunan bingkai gambar secara berkesinambungan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            st.info("📸 Ambil foto berulang sebanyak 20x untuk membangun buffer sekuensial kata isyarat.")
            cam_kata = st.camera_input("📷 Capture frame isyarat tangan", key="cam_kata")
        with c2:
            st.markdown("##### 📊 ANALISIS BINGKAI TEMPORAL")
            buffer_progress = st.empty()
            kata_lbl        = st.empty()

        # Simpan buffer frame di session_state
        if "kata_frame_buffer" not in st.session_state:
            st.session_state.kata_frame_buffer = []

        if cam_kata is not None:
            img_kata = Image.open(cam_kata).resize((96, 96))
            frame_arr = np.array(img_kata, dtype=np.float32) / 255.0

            # Tambah ke buffer, maksimal 20 frame
            st.session_state.kata_frame_buffer.append(frame_arr)
            if len(st.session_state.kata_frame_buffer) > 20:
                st.session_state.kata_frame_buffer.pop(0)

            current_len = len(st.session_state.kata_frame_buffer)
            buffer_progress.progress(float(current_len / 20), text=f"Stabilitas Buffer Isyarat: {current_len}/20 Frames Packed")

            if current_len == 20 and video_model is not None and video_class_names:
                frames_np  = np.array(st.session_state.kata_frame_buffer, dtype=np.float32)
                video_arr  = np.expand_dims(frames_np, axis=0)
                predictions = video_model.predict(video_arr, verbose=0)
                pred_class  = np.argmax(predictions[0])
                label       = video_class_names[pred_class]
                conf        = float(predictions[0][pred_class] * 100)

                kata_lbl.markdown(f"""
                <div class="kpi-box" style="border-left-color: #10b981;">
                    <div class="kpi-title">Terjemahan Frasa/Kata</div>
                    <div class="kpi-value" style="color: #059669;">"{label.upper()}"</div>
                    <small style="color: #64748b;">Akurasi Sekuensial: {conf:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            elif video_model is None:
                kata_lbl.error("❌ Model video tidak tersedia.")
            else:
                kata_lbl.warning("Menyelaraskan data sekuens... Pertahankan posisi tangan Anda di depan kamera.")

        # Tombol reset buffer
        if st.button("🔄 Reset Buffer Frame", key="reset_kata"):
            st.session_state.kata_frame_buffer = []
            buffer_progress.empty()
            kata_lbl.info("Buffer direset. Mulai capture ulang.")

        if cam_kata is None and not st.session_state.kata_frame_buffer:
            kata_lbl.info("Aktifkan toggle di atas untuk mulai memproses data gestur berbasis waktu secara interaktif.")
