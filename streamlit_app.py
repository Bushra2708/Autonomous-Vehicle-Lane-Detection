import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os

# ---------------------------------------------------
# FIX FOR OLD .H5 MODEL (IMPORTANT)
# ---------------------------------------------------
keras.config.enable_unsafe_deserialization()

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Lane Detection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CUSTOM CSS (UNCHANGED)
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #050816;
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #050816, #0f172a, #111827);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    height: 45px;
    font-size: 15px;
    font-weight: 500;
    color: white;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 14px;
    height: 45px;
    font-size: 15px;
    font-weight: 500;
    color: white;
    background: linear-gradient(135deg, #059669, #10b981);
}

footer, #MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL (FIXED)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_lane_detection_model.h5")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model

model = load_model()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="main-title">Autonomous Lane Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a road image to detect lanes</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    original = image.copy()

    # PREPROCESS
    resized = cv2.resize(image, (256, 128))
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)

    # PREDICTION
    with st.spinner("Processing image..."):
        prediction = model.predict(input_image)

    mask = prediction[0].squeeze()
    mask = (mask > 0.5).astype(np.uint8)

    # RESIZE MASK
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # OVERLAY
    lane_mask = np.zeros_like(original)
    lane_mask[:, :, 1] = 255
    lane_mask[:, :, 2] = 255

    overlay = np.where(
        mask[:, :, np.newaxis] == 1,
        cv2.addWeighted(original, 0.4, lane_mask, 0.9, 0),
        original
    ).astype(np.uint8)

    # DISPLAY
    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Original", use_container_width=True)

    with col2:
        st.image(overlay, caption="Lane Detection", use_container_width=True)

    # DOWNLOAD
    result_image = Image.fromarray(overlay)

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")

    st.download_button(
        label="Download Result",
        data=buf.getvalue(),
        file_name="lane_detection_result.png",
        mime="image/png"
    )

st.markdown('</div>', unsafe_allow_html=True)