"""
====================================================================
APLIKASI KLASIFIKASI GESTUR TANGAN BISINDO
Bahasa Isyarat Indonesia (BISINDO) - Stable Version
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

# ============================================
# KONFIGURASI HALAMAN
# ============================================

st.set_page_config(
    page_title="BISINDO - Bahasa Isyarat Indonesia",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS KUSTOM
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
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .result-box h2 {
        font-size: 2.5rem;
        margin: 0;
    }
    .result-box .confidence {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .result-box-video {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .camera-container {
        border-radius: 15px;
        overflow: hidden;
        border: 3px solid #667eea;
        background: #000;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-active {
        background: #28a745;
        color: white;
    }
    .status-inactive {
        background: #dc3545;
        color: white;
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
# LOAD MODEL & CLASS NAMES
# ============================================

@st.cache_resource
def load_models():
    """Load model dan class names dari file"""
    try:
        model_path = "models"
        
        # Load model gambar
        image_model = tf.keras.models.load_model(f"{model_path}/image_model.h5")
        
        # Load model video
        video_model = tf.keras.models.load_model(f"{model_path}/video_model.h5")
        
        # Load class names
        with open(f"{model_path}/image_class_names.pkl", "rb") as f:
            image_class_names = pickle.load(f)
        
        with open(f"{model_path}/video_class_names.pkl", "rb") as f:
            video_class_names = pickle.load(f)
        
        return image_model, video_model, image_class_names, video_class_names
    
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None

# ============================================
# FUNGSI PREPROCESSING
# ============================================

def preprocess_image(image):
    """Preprocessing gambar untuk MobileNetV2"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_video(video_file, max_frames=20, img_size=96):
    """Preprocessing video untuk MobileNetV2 + LSTM"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    frames = []
    
    if not cap.isOpened():
        os.remove(tmp_path)
        return None
    
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
        return None
    
    indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    
    for idx in indices:
        if idx < total_frames:
            frame = all_frames[idx]
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
    
    video_array = np.array(frames, dtype=np.float32)
    video_array = np.expand_dims(video_array, axis=0)
    
    return video_array

# ============================================
# FUNGSI PREDIKSI
# ============================================

def predict_image(model, image, class_names):
    """Prediksi gambar"""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return class_names[pred_class], confidence, predictions[0]

def predict_video(model, video_file, class_names):
    """Prediksi video"""
    processed = preprocess_video(video_file)
    if processed is None:
        return None, 0, None
    
    predictions = model.predict(processed, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return class_names[pred_class], confidence, predictions[0]

def predict_frame_gambar(model, frame, class_names):
    """Prediksi frame untuk model gambar (Abjad) - STABIL"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        return class_names[pred_class], confidence, predictions[0]
    except Exception as e:
        return "Error", 0, None

def predict_frame_kata(model, frame, class_names, max_frames=20, img_size=96):
    """Prediksi frame untuk model video (Kata) - STABIL"""
    # Inisialisasi buffer di session state
    if 'buffer_kata' not in st.session_state:
        st.session_state.buffer_kata = []
    
    try:
        # Proses frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Tambahkan ke buffer
        st.session_state.buffer_kata.append(frame_normalized)
        
        # Jika buffer terlalu besar, potong
        if len(st.session_state.buffer_kata) > max_frames:
            st.session_state.buffer_kata = st.session_state.buffer_kata[-max_frames:]
        
        # Prediksi jika buffer penuh
        if len(st.session_state.buffer_kata) >= max_frames:
            frames = np.array(st.session_state.buffer_kata[-max_frames:])
            frames = np.expand_dims(frames, axis=0)
            
            predictions = model.predict(frames, verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            return class_names[pred_class], confidence, predictions[0]
        else:
            return None, 0, None
    except Exception as e:
        return "Error", 0, None

def reset_kata_buffer():
    """Reset buffer untuk kata"""
    if 'buffer_kata' in st.session_state:
        st.session_state.buffer_kata = []

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <h3>📋 Informasi</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Fitur
    
    - 📸 **Gambar**: Upload gambar gestur abjad
    - 🎬 **Video**: Upload video gestur kata
    - 📷 **Kamera Abjad**: Deteksi real-time abjad
    - 📷 **Kamera Kata**: Deteksi real-time kata
    
    ### 📊 Dataset
    
    - **Abjad**: 26 huruf (A-Z)
    - **Kata**: 24 kata BISINDO
    
    ### 🧠 Model
    
    - **Gambar**: MobileNetV2 (Transfer Learning)
    - **Video**: MobileNetV2 + LSTM
    """)
    
    st.divider()
    
    if st.button("🔄 Reset Buffer Kata", use_container_width=True):
        reset_kata_buffer()
        st.success("✅ Buffer direset!")
    
    st.caption("© 2024 BISINDO Classification")

# ============================================
# MAIN APP
# ============================================

def main():
    # Load model
    with st.spinner("⏳ Memuat model..."):
        image_model, video_model, image_class_names, video_class_names = load_models()
    
    if image_model is None or video_model is None:
        st.warning("⚠️ Model tidak ditemukan. Pastikan file model ada di folder 'models/'")
        st.info("📁 Struktur folder yang dibutuhkan:")
        st.code("""
        project/
        ├── app.py
        ├── models/
        │   ├── image_model.h5
        │   ├── video_model.h5
        │   ├── image_class_names.pkl
        │   └── video_class_names.pkl
        └── requirements.txt
        """)
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Gambar (Abjad)",
        "🎬 Video (Kata)",
        "📷 Kamera (Abjad)",
        "📷 Kamera (Kata)"
    ])
    
    # ============================================
    # TAB 1: GAMBAR (ABJAD)
    # ============================================
    
    with tab1:
        st.markdown("""
        <div class="card">
            <h3>📸 Upload Gambar</h3>
            <p>Upload gambar gestur tangan abjad (A-Z)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file gambar...",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=False,
            key="image_uploader"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="🖼️ Gambar yang diupload", use_container_width=True)
        
        with col2:
            if uploaded_file is not None:
                if st.button("🔍 Prediksi Gambar", type="primary", use_container_width=True):
                    with st.spinner("⏳ Memproses prediksi..."):
                        start_time = time.time()
                        label, confidence, all_probs = predict_image(
                            image_model, image, image_class_names
                        )
                        elapsed = time.time() - start_time
                        
                        st.markdown(f"""
                        <div class="result-box fade-in">
                            <h2>🎯 {label}</h2>
                            <p class="confidence">Confidence: {confidence:.2f}%</p>
                            <p style="font-size:0.8rem; opacity:0.7;">⏱️ {elapsed:.2f} detik</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.subheader("📊 Top 5 Prediksi")
                        top_indices = np.argsort(all_probs)[-5:][::-1]
                        for idx in top_indices:
                            prob = float(all_probs[idx] * 100)
                            col_progress, col_label = st.columns([3, 1])
                            with col_progress:
                                st.progress(prob / 100)
                            with col_label:
                                st.write(f"{image_class_names[idx]} {prob:.1f}%")
            else:
                st.info("⬅️ Upload gambar untuk memulai prediksi")
    
    # ============================================
    # TAB 2: VIDEO (KATA)
    # ============================================
    
    with tab2:
        st.markdown("""
        <div class="card">
            <h3>🎬 Upload Video</h3>
            <p>Upload video gestur tangan kata BISINDO</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file video...",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
            key="video_uploader"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                cap = cv2.VideoCapture(tmp_path)
                ret, frame = cap.read()
                cap.release()
                os.remove(tmp_path)
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="🎬 Sample Frame Video", use_container_width=True)
                else:
                    st.warning("⚠️ Tidak dapat membaca video")
        
        with col2:
            if uploaded_file is not None:
                if st.button("🔍 Prediksi Video", type="primary", use_container_width=True):
                    uploaded_file.seek(0)
                    
                    with st.spinner("⏳ Memproses prediksi..."):
                        start_time = time.time()
                        label, confidence, all_probs = predict_video(
                            video_model, uploaded_file, video_class_names
                        )
                        elapsed = time.time() - start_time
                        
                        if label is None:
                            st.error("❌ Gagal memproses video!")
                        else:
                            st.markdown(f"""
                            <div class="result-box-video fade-in">
                                <h2>🎯 {label}</h2>
                                <p class="confidence">Confidence: {confidence:.2f}%</p>
                                <p style="font-size:0.8rem; opacity:0.7;">⏱️ {elapsed:.2f} detik</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("📊 Top 5 Prediksi")
                            top_indices = np.argsort(all_probs)[-5:][::-1]
                            for idx in top_indices:
                                prob = float(all_probs[idx] * 100)
                                col_progress, col_label = st.columns([3, 1])
                                with col_progress:
                                    st.progress(prob / 100)
                                with col_label:
                                    st.write(f"{video_class_names[idx]} {prob:.1f}%")
            else:
                st.info("⬅️ Upload video untuk memulai prediksi")
    
    # ============================================
    # TAB 3: KAMERA ABJAD (REAL-TIME)
    # ============================================
    
    with tab3:
        st.markdown("""
        <div class="card">
            <h3>📷 Deteksi Real-Time Abjad dengan Kamera</h3>
            <p>Deteksi gestur abjad (A-Z) secara langsung dari kamera</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_cam, col_result = st.columns([2, 1])
        
        with col_cam:
            st.markdown("### 📷 Live Camera")
            
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            col_start, col_stop = st.columns(2)
            with col_start:
                start = st.button("▶️ Start Camera", type="primary", use_container_width=True, key="start_abjad")
            with col_stop:
                stop = st.button("⏹️ Stop Camera", use_container_width=True, key="stop_abjad")
        
        with col_result:
            st.markdown("### 🎯 Hasil Deteksi")
            result_placeholder = st.empty()
        
        # State
        if 'cam_abjad_running' not in st.session_state:
            st.session_state.cam_abjad_running = False
            st.session_state.cam_abjad_label = "Menunggu..."
            st.session_state.cam_abjad_conf = 0
        
        if start:
            st.session_state.cam_abjad_running = True
            st.session_state.cam_abjad_label = "Memulai..."
            st.session_state.cam_abjad_conf = 0
        
        if stop:
            st.session_state.cam_abjad_running = False
            video_placeholder.empty()
            result_placeholder.empty()
            status_placeholder.empty()
        
        if st.session_state.cam_abjad_running:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Kamera tidak tersedia!")
                    st.session_state.cam_abjad_running = False
                else:
                    status_placeholder.info("🟢 Kamera aktif...")
                    
                    frame_count = 0
                    while st.session_state.cam_abjad_running:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Prediksi setiap 5 frame
                        if frame_count % 5 == 0:
                            label, conf, _ = predict_frame_gambar(
                                image_model, frame, image_class_names
                            )
                            if label != "Error":
                                st.session_state.cam_abjad_label = label
                                st.session_state.cam_abjad_conf = conf
                        
                        # Anotasi frame
                        label_text = f"{st.session_state.cam_abjad_label} ({st.session_state.cam_abjad_conf:.1f}%)"
                        cv2.putText(frame, label_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Tampilkan frame
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        except Exception:
                            pass
                        
                        # Update hasil
                        if frame_count % 5 == 0:
                            try:
                                result_placeholder.markdown(f"""
                                <div class="result-box fade-in">
                                    <h2 style="font-size:2rem;">🎯 {st.session_state.cam_abjad_label}</h2>
                                    <p class="confidence">Confidence: {st.session_state.cam_abjad_conf:.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception:
                                pass
                        
                        time.sleep(0.03)
                    
                    cap.release()
                    video_placeholder.empty()
                    result_placeholder.empty()
                    status_placeholder.empty()
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.cam_abjad_running = False
                video_placeholder.empty()
                result_placeholder.empty()
                status_placeholder.empty()
        else:
            status_placeholder.info("⏸️ Klik 'Start Camera' untuk memulai")
            result_placeholder.info("📷 Mulai kamera untuk melihat hasil")
    
    # ============================================
    # TAB 4: KAMERA KATA (REAL-TIME)
    # ============================================
    
    with tab4:
        st.markdown("""
        <div class="card">
            <h3>📷 Deteksi Real-Time Kata dengan Kamera</h3>
            <p>Deteksi gestur kata BISINDO secara langsung dari kamera</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_cam, col_result = st.columns([2, 1])
        
        with col_cam:
            st.markdown("### 📷 Live Camera")
            
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            col_start, col_stop = st.columns(2)
            with col_start:
                start = st.button("▶️ Start Camera", type="primary", use_container_width=True, key="start_kata")
            with col_stop:
                stop = st.button("⏹️ Stop Camera", use_container_width=True, key="stop_kata")
        
        with col_result:
            st.markdown("### 🎯 Hasil Deteksi")
            result_placeholder = st.empty()
        
        # State
        if 'cam_kata_running' not in st.session_state:
            st.session_state.cam_kata_running = False
            st.session_state.cam_kata_label = "Menunggu..."
            st.session_state.cam_kata_conf = 0
        
        if 'buffer_kata' not in st.session_state:
            st.session_state.buffer_kata = []
        
        if start:
            st.session_state.cam_kata_running = True
            st.session_state.cam_kata_label = "Mengisi buffer..."
            st.session_state.cam_kata_conf = 0
            st.session_state.buffer_kata = []
        
        if stop:
            st.session_state.cam_kata_running = False
            st.session_state.buffer_kata = []
            video_placeholder.empty()
            result_placeholder.empty()
            status_placeholder.empty()
        
        if st.session_state.cam_kata_running:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Kamera tidak tersedia!")
                    st.session_state.cam_kata_running = False
                else:
                    status_placeholder.info("🟢 Kamera aktif...")
                    
                    frame_count = 0
                    while st.session_state.cam_kata_running:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Prediksi dengan buffer
                        label, conf, _ = predict_frame_kata(
                            video_model, frame, video_class_names
                        )
                        
                        if label is not None and label != "Error":
                            st.session_state.cam_kata_label = label
                            st.session_state.cam_kata_conf = conf
                        elif label == "Error":
                            st.session_state.cam_kata_label = "Error"
                            st.session_state.cam_kata_conf = 0
                        else:
                            # Sedang mengisi buffer
                            buffer_size = len(st.session_state.buffer_kata)
                            st.session_state.cam_kata_label = f"Mengisi buffer... ({buffer_size}/20)"
                            st.session_state.cam_kata_conf = 0
                        
                        # Anotasi frame
                        buffer_size = len(st.session_state.buffer_kata)
                        label_text = f"{st.session_state.cam_kata_label} ({st.session_state.cam_kata_conf:.1f}%)"
                        cv2.putText(frame, label_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Buffer: {buffer_size}/20", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Tampilkan frame
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        except Exception:
                            pass
                        
                        # Update hasil
                        try:
                            if st.session_state.cam_kata_conf > 0:
                                result_placeholder.markdown(f"""
                                <div class="result-box-video fade-in">
                                    <h2 style="font-size:2rem;">🎯 {st.session_state.cam_kata_label}</h2>
                                    <p class="confidence">Confidence: {st.session_state.cam_kata_conf:.1f}%</p>
                                    <p style="font-size:0.8rem; opacity:0.7;">Buffer: {buffer_size}/20 ✅</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                result_placeholder.markdown(f"""
                                <div class="result-box fade-in" style="background: #6c757d;">
                                    <h2 style="font-size:1.5rem;">⏳ {st.session_state.cam_kata_label}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception:
                            pass
                        
                        time.sleep(0.03)
                    
                    cap.release()
                    video_placeholder.empty()
                    result_placeholder.empty()
                    status_placeholder.empty()
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.cam_kata_running = False
                video_placeholder.empty()
                result_placeholder.empty()
                status_placeholder.empty()
        else:
            status_placeholder.info("⏸️ Klik 'Start Camera' untuk memulai")
            result_placeholder.info("📷 Mulai kamera untuk melihat hasil")

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()
