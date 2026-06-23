# ============================================
# VERSI BARU - PAKAI FETCH
# ============================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import base64
import io
import json

# ============================================
# HTML REAL-TIME DENGAN FETCH
# ============================================

def realtime_html():
    return """
    <div style="text-align: center; margin: 10px 0;">
        <video id="video" width="100%" height="auto" autoplay style="max-height: 400px; background: #000; border-radius: 10px;"></video>
        <br><br>
        <div id="result" style="font-size: 28px; font-weight: bold; min-height: 50px; padding: 15px; background: #f8f9fa; border-radius: 10px; margin-top: 10px;">
            🎯 Menunggu...
        </div>
        <div id="confidence" style="font-size: 18px; color: #28a745;">Confidence: 0%</div>
        <br>
        <button id="startBtn" style="background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer;">▶️ Start</button>
        <button id="stopBtn" style="background: #dc3545; color: white; padding: 12px 30px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer;">⏹️ Stop</button>
        <canvas id="canvas" style="display: none;"></canvas>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const resultDiv = document.getElementById('result');
        const confidenceDiv = document.getElementById('confidence');
        
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        
        function sendToStreamlit(imageData) {
            // Kirim data ke Streamlit melalui POST
            fetch(window.location.origin + '/_stcore/upload_file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData
                })
            }).catch(err => console.log('Send error:', err));
            
            // Alternatif: kirim melalui query params
            window.location.href = window.location.href.split('?')[0] + 
                '?img=' + encodeURIComponent(imageData);
        }
        
        function captureAndPredict() {
            if (!isRunning) return;
            
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Kirim ke Streamlit
            sendToStreamlit(imageData);
            
            resultDiv.innerHTML = '⏳ Memproses...';
            resultDiv.style.color = '#667eea';
        }
        
        function startCamera() {
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
                .then(mediaStream => {
                    stream = mediaStream;
                    video.srcObject = mediaStream;
                    video.play();
                    isRunning = true;
                    resultDiv.innerHTML = '🟢 Camera running...';
                    resultDiv.style.color = '#28a745';
                    
                    if (intervalId) clearInterval(intervalId);
                    intervalId = setInterval(captureAndPredict, 500);
                })
                .catch(err => {
                    resultDiv.innerHTML = '❌ Error: ' + err.message;
                    resultDiv.style.color = 'red';
                });
        }
        
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
            resultDiv.innerHTML = '⏸️ Stopped';
            resultDiv.style.color = '#6c757d';
        }
        
        document.getElementById('startBtn').addEventListener('click', startCamera);
        document.getElementById('stopBtn').addEventListener('click', stopCamera);
    </script>
    """

# ============================================
# MAIN APP
# ============================================

def main():
    st.set_page_config(page_title="BISINDO - Kamera", layout="wide")
    
    st.title("🖐️ BISINDO - Deteksi Gestur")
    
    # Load model
    @st.cache_resource
    def load_model():
        try:
            model = tf.keras.models.load_model("models/image_model.h5")
            with open("models/image_class_names.pkl", "rb") as f:
                class_names = pickle.load(f)
            return model, class_names
        except Exception as e:
            st.error(f"❌ Gagal load model: {e}")
            return None, None
    
    model, class_names = load_model()
    
    if model is None:
        st.error("Model tidak ditemukan!")
        return
    
    # Tampilkan kamera
    st.components.v1.html(realtime_html(), height=500)
    
    # Proses data dari query params
    query_params = st.query_params
    if 'img' in query_params:
        try:
            img_data = query_params['img']
            img_b64 = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Prediksi
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array, verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Tampilkan hasil
            st.success(f"🎯 Hasil: {class_names[pred_class]}")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Clear query params
            st.query_params.clear()
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
