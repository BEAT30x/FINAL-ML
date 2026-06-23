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
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ============================================
# 1. KONFIGURASI HALAMAN & TEMA UTAMA
# ============================================
st.set_page_config(
    page_title="BISINDO Intelligence Translator",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

        image_model       = None
        video_model       = None
        image_class_names = None
        video_class_names = None

        img_path  = os.path.join(BASE_DIR, "image_model.h5")
        vid_path  = os.path.join(BASE_DIR, "video_model.h5")
        ipkl_path = os.path.join(BASE_DIR, "image_class_names.pkl")
        vpkl_path = os.path.join(BASE_DIR, "video_class_names.pkl")

        def is_valid_h5(path):
            if not os.path.exists(path):
                return False, "File tidak ditemukan"
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 0.1:
                return False, f"File terlalu kecil ({size_mb:.2f} MB) — kemungkinan Git LFS pointer"
            return True, f"{size_mb:.1f} MB"

        img_valid, img_info = is_valid_h5(img_path)
        vid_valid, vid_info = is_valid_h5(vid_path)

        if img_valid:
            try:
                image_model = tf.keras.models.load_model(img_path)
            except Exception as e:
                st.error(f"❌ Gagal load image_model.h5 ({img_info}): {e}")
        else:
            st.error(f"❌ image_model.h5 — {img_info}")

        if vid_valid:
            try:
                video_model = tf.keras.models.load_model(vid_path)
            except Exception as e:
                st.warning(f"⚠️ Gagal load video_model.h5 ({vid_info}): {e}")
        else:
            st.warning(f"⚠️ video_model.h5 — {vid_info}")

        try:
            with open(ipkl_path, "rb") as f:
                image_class_names = pickle.load(f)
        except:
            image_class_names = [chr(65 + i) for i in range(26)]

        try:
            with open(vpkl_path, "rb") as f:
                video_class_names = pickle.load(f)
        except:
            video_class_names = []

        return image_model, video_model, image_class_names, video_class_names

    except Exception as e:
        st.error(f"❌ Error tidak terduga: {e}")
        return None, None, None, None


image_model, video_model, image_class_names, video_class_names = load_models()

# ============================================
# 4. RTC CONFIGURATION (STUN server publik)
# ============================================
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ============================================
# 5. VIDEO PROCESSOR — ABJAD (per-frame)
# ============================================
class AbjadProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_label = "-"
        self.result_conf  = 0.0

    def recv(self, frame):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img_bgr  = frame.to_ndarray(format="bgr24")
        img_bgr  = cv2.flip(img_bgr, 1)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil  = Image.fromarray(img_rgb).resize((224, 224))
        arr      = np.array(img_pil, dtype=np.float32)
        arr      = preprocess_input(arr)
        arr      = np.expand_dims(arr, axis=0)

        probs      = image_model.predict(arr, verbose=0)[0]
        pred_idx   = np.argmax(probs)
        self.result_label = image_class_names[pred_idx]
        self.result_conf  = float(probs[pred_idx] * 100)

        label_text = f"{self.result_label}  {self.result_conf:.1f}%"
        cv2.rectangle(img_bgr, (0, 0), (300, 50), (79, 70, 229), -1)
        cv2.putText(img_bgr, label_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), format="rgb24")


# ============================================
# 6. VIDEO PROCESSOR — KATA (buffer 20 frame)
# ============================================
class KataProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer       = deque(maxlen=20)
        self.result_label = "-"
        self.result_conf  = 0.0
        self.frame_count  = 0

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_bgr = cv2.flip(img_bgr, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.frame_count += 1
        if self.frame_count % 3 == 0:
            resized = cv2.resize(img_rgb, (96, 96))
            self.buffer.append(resized.astype(np.float32) / 255.0)

        buf_len = len(self.buffer)

        if buf_len == 20 and video_model is not None and video_class_names:
            frames_np   = np.array(list(self.buffer), dtype=np.float32)
            video_arr   = np.expand_dims(frames_np, axis=0)
            preds       = video_model.predict(video_arr, verbose=0)[0]
            pred_class  = np.argmax(preds)
            self.result_label = video_class_names[pred_class]
            self.result_conf  = float(preds[pred_class] * 100)
            self.buffer.clear()

        bar_width = int((buf_len / 20) * 300)
        cv2.rectangle(img_bgr, (0, 0), (320, 55), (15, 23, 42), -1)
        cv2.rectangle(img_bgr, (5, 35), (5 + bar_width, 48), (16, 185, 129), -1)
        label_text = f"{self.result_label}  {self.result_conf:.1f}%"
        cv2.putText(img_bgr, label_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img_bgr, f"Buffer: {buf_len}/20", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return av.VideoFrame.from_ndarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), format="rgb24")


# ============================================
# 7. PREPROCESSING PIPELINES
# ============================================
def preprocess_image(image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image     = image.resize((224, 224))
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
    frames  = []
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    for idx in indices:
        if idx < total_frames:
            f = all_frames[idx]
            f = cv2.resize(f, (img_size, img_size))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f.astype(np.float32) / 255.0)
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
    return np.expand_dims(np.array(frames, dtype=np.float32), axis=0)


# ============================================
# 8. SIDEBAR
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
if image_model is None:
    st.error("🚨 Master Model File (`.h5` / `.pkl`) tidak ditemukan.")
    st.info("📁 Pastikan `image_model.h5` dan file pkl ada di root folder repository.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Unggah Citra (Abjad)",
        "📂 Unggah Video (Kata)",
        "⚡ Live Stream (Abjad/Huruf)",
        "⚡ Live Stream (Kata/Kalimat)"
    ])

    # ----------------------------------------------------------------
    # TAB 1: IMAGE UPLOADER
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
                        probs     = image_model.predict(processed, verbose=0)[0]
                        pred_idx  = np.argmax(probs)
                        label     = image_class_names[pred_idx]
                        confidence = probs[pred_idx] * 100
                        st.markdown(f"""
                        <div class="kpi-box">
                            <div class="kpi-title">Hasil Klasifikasi Dominan</div>
                            <div class="kpi-value">Abjad: {label}</div>
                            <div style="color:#166534;font-size:0.9rem;font-weight:500;">Tingkat Kepercayaan: {confidence:.2f}%</div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown("<br><b>Distribusi Keyakinan Model:</b>", unsafe_allow_html=True)
                        top5 = np.argsort(probs)[-5:][::-1]
                        for idx in top5:
                            st.text(f"Kelas [{image_class_names[idx]}]: {probs[idx]*100:.1f}%")
                            st.progress(float(probs[idx]))
            else:
                st.info("💡 Hubungkan dokumen gambar di panel kiri untuk mengaktifkan kalkulasi model.")

    # ----------------------------------------------------------------
    # TAB 2: VIDEO UPLOADER
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
                                probs     = video_model.predict(processed_vid, verbose=0)[0]
                                pred_idx  = np.argmax(probs)
                                label     = video_class_names[pred_idx] if video_class_names else f"Kelas {pred_idx}"
                                confidence = probs[pred_idx] * 100
                                st.markdown(f"""
                                <div class="kpi-box">
                                    <div class="kpi-title">Kata Isyarat Terjemahan</div>
                                    <div class="kpi-value">"{label.upper()}"</div>
                                    <div style="color:#166534;font-size:0.9rem;font-weight:500;">Tingkat Konfidensi Sekuensial: {confidence:.2f}%</div>
                                </div>""", unsafe_allow_html=True)
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
    # TAB 3: LIVE STREAM REAL-TIME — ABJAD (FIX: hapus while loop)
    # ----------------------------------------------------------------
    with tab3:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Abjad</h3><p>Kamera berjalan real-time dan menerjemahkan abjad setiap frame secara otomatis.</p></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            ctx_abjad = webrtc_streamer(
                key="abjad-stream",
                video_processor_factory=AbjadProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        with c2:
            st.markdown("##### 📊 LOG PREDIKSI SISTEM")
            lbl_box  = st.empty()
            conf_bar = st.empty()
            hist_box = st.empty()

            if "abjad_history" not in st.session_state:
                st.session_state.abjad_history = []

            # FIX: Ganti while loop dengan st.rerun() untuk menghindari DOM crash
            if ctx_abjad.video_processor:
                label = ctx_abjad.video_processor.result_label
                conf  = ctx_abjad.video_processor.result_conf

                lbl_box.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-title">Karakter Terbaca</div>
                    <div class="kpi-value" style="color:#4f46e5;">{label}</div>
                </div>""", unsafe_allow_html=True)
                conf_bar.progress(conf / 100, text=f"Tingkat Kepastian Akurasi: {conf:.1f}%")

                if label != "-":
                    if (not st.session_state.abjad_history or
                            st.session_state.abjad_history[-1] != label):
                        st.session_state.abjad_history.append(label)
                    if len(st.session_state.abjad_history) > 15:
                        st.session_state.abjad_history.pop(0)

                hist_box.markdown("**Histori:** " + " → ".join(st.session_state.abjad_history))

                time.sleep(0.1)
                st.rerun()
            else:
                lbl_box.info("Klik **START** pada panel kamera untuk mengaktifkan deteksi real-time.")

    # ----------------------------------------------------------------
    # TAB 4: LIVE STREAM REAL-TIME — KATA (FIX: hapus while loop)
    # ----------------------------------------------------------------
    with tab4:
        st.markdown('<div class="pro-card"><h3>Live Tracking Model: Kata (Sistem Sekuensial)</h3><p>Model mengumpulkan 20 frame otomatis dari kamera secara berkesinambungan.</p></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            ctx_kata = webrtc_streamer(
                key="kata-stream",
                video_processor_factory=KataProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        with c2:
            st.markdown("##### 📊 ANALISIS BINGKAI TEMPORAL")
            buf_bar  = st.empty()
            kata_box = st.empty()

            # FIX: Ganti while loop dengan st.rerun() untuk menghindari DOM crash
            if ctx_kata.video_processor:
                label   = ctx_kata.video_processor.result_label
                conf    = ctx_kata.video_processor.result_conf
                buf_len = len(ctx_kata.video_processor.buffer)

                buf_bar.progress(
                    float(buf_len / 20),
                    text=f"Stabilitas Buffer Isyarat: {buf_len}/20 Frames Packed"
                )

                if label != "-":
                    kata_box.markdown(f"""
                    <div class="kpi-box" style="border-left-color:#10b981;">
                        <div class="kpi-title">Terjemahan Frasa/Kata</div>
                        <div class="kpi-value" style="color:#059669;">"{label.upper()}"</div>
                        <small style="color:#64748b;">Akurasi Sekuensial: {conf:.1f}%</small>
                    </div>""", unsafe_allow_html=True)
                else:
                    kata_box.warning("Menyelaraskan data sekuens... Pertahankan posisi tangan Anda di depan kamera.")

                time.sleep(0.1)
                st.rerun()
            else:
                kata_box.info("Klik **START** pada panel kamera untuk mulai memproses data gestur berbasis waktu.")
