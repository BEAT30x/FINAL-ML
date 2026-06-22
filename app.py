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
    /* Mengubah font global dan background aplikasi */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc; /* Off-white yang nyaman di mata */
    }
    
    /* Header Profesional Menyerupai Dashboard SaaS */
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
        color: #0f172a; /* Slate 900 */
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-title-section p {
        font-size: 0.95rem;
        color: #64748b; /* Slate 500 */
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
    
    /* Desain Kartu (Cards) Komponen */
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

    /* Hasil Prediksi Utama (KPI Metric Display) */
    .kpi-box {
        background: #fdfdfd;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5; /* Aksen Indigo */
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
    
    /* Kustomisasi Tab Streamlit agar Lebih Clean */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        color: #64748b;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4f46e5 !important;
        font-weight: 600;
    }
    
    /* Mempercantik Frame Kamera Langsung */
    .video-stream-container img {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. APPLICATION HEADER (SaaS Style)
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
# 3. LOAD CORE MODELS (Optimized Caching)
# ============================================
@st.cache_resource
def load_models():
    try:
        model_path = "models"
        image_model = tf.keras.models.load_model(f"{model_path}/image_model.h5")
        video_model = tf.keras.models.load_model(f"{model_path}/video_model.h5")
        
        with open(f"{model_path}/image_class_names.pkl", "rb") as f:
            image_class_names = pickle.load(f)
        with open(f"{model_path}/video_class_names.pkl", "rb") as f:
            video_class_names = pickle.load(f)
        
        return image_model, video_model, image_class_names, video_class_names
    except Exception as e:
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
# 6. MAIN CONTROLLER & MAIN NAVIGATION Tabs
# ============================================
if image_model is None or video_model is None:
    st.error("🚨 Master Model File (`.h5` / `.pkl`) tidak ditemukan di folder `models/`. Harap unggah model telebih dahulu.")
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
    # TAB 3: LIVE STREAM CAMERA (ABJAD)
    # ----------------------------------------------------------------
    with tab3:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Abjad</h3><p>Gunakan kamera untuk menerjemahkan abjad secara instan. Matriks hasil diperbarui otomatis di sisi kanan.</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            run_abjad = st.toggle("Aktifkan Jalur Kamera Abjad", key="tg_abjad")
            frame_window = st.empty()
        with c2:
            st.markdown("##### 📊 LOG PREDIKSI SISTEM")
            lbl_placeholder = st.empty()
            conf_placeholder = st.empty()
            
        if run_abjad:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Sistem gagal terhubung ke perangkat keras Kamera.")
            else:
                while run_abjad:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # UX Lebih Natural: Efek Cermin (Mirroring)
                    frame = cv2.flip(frame, 1)
                    label, conf = predict_frame_gambar(image_model, frame, image_class_names)
                    
                    # Render Informasi ke Widget Clean Streamlit (Bukan Teks OpenCV)
                    lbl_placeholder.markdown(f"""
                    <div class="kpi-box">
                        <div class="kpi-title">Karakter Terbaca</div>
                        <div class="kpi-value" style="color: #4f46e5;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # FIXED: Konversi eksplisit np.float32 ke Python standard float
                    conf_placeholder.progress(float(conf / 100), text=f"Tingkat Kepastian Akurasi: {conf:.1f}%")
                    
                    # Output Video tanpa OSD kaku
                    frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_disp, channels="RGB", use_container_width=True)
                cap.release()
        else:
            frame_window.info("Toggle sakelar di atas untuk mengaktifkan modul jepretan kamera real-time.")

    # ----------------------------------------------------------------
    # TAB 4: LIVE STREAM CAMERA (KATA)
    # ----------------------------------------------------------------
    with tab4:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Kata (Sistem Sekuensial)</h3><p>Model mengumpulkan 20 runtunan bingkai gambar secara berkesinbaungan untuk menyimpulkan sebuah kata tunggal.</p></div>', unsafe_allow_html=True)
        
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
                    # FIXED: Konversi eksplisit np.float32 ke Python standard float
                    buffer_progress.progress(float(current_len / 20), text=f"Stabilitas Buffer Isyarat: {current_len}/20 Frames Packed")
                    
                    if current_len == 20:
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
                cap.release()
        else:
            frame_window_kata.info("Aktifkan toggle di atas untuk mulai memproses data gestur berbasis waktu secara interaktif.")
