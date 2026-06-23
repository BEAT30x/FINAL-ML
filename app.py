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
# 3. LOAD CORE MODELS - DENGAN CEK VALIDITAS FILE
# ============================================
@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Cek file di berbagai lokasi
        locations = [
            os.path.join(BASE_DIR, "models"),
            BASE_DIR,
            os.path.join(BASE_DIR, "model"),
        ]
        
        for loc in locations:
            image_h5 = os.path.join(loc, "image_model.h5")
            video_h5 = os.path.join(loc, "video_model.h5")
            image_pkl = os.path.join(loc, "image_class_names.pkl")
            video_pkl = os.path.join(loc, "video_class_names.pkl")
            
            # CEK UKURAN FILE - file yang valid minimal 1MB
            if os.path.exists(image_h5) and os.path.exists(video_h5):
                img_size = os.path.getsize(image_h5) / (1024 * 1024)
                vid_size = os.path.getsize(video_h5) / (1024 * 1024)
                
                if img_size < 1 or vid_size < 1:
                    st.warning(f"⚠️ File model di {loc} terlalu kecil ({img_size:.2f}MB / {vid_size:.2f}MB), mungkin rusak!")
                    continue
                
                try:
                    st.info(f"⏳ Loading model dari: {loc}...")
                    image_model = tf.keras.models.load
