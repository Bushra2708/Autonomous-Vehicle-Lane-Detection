import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Autonomous Lane Detection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #050816;
    color: white;
}

.stApp {
    background: linear-gradient(
        135deg,
        #050816,
        #0f172a,
        #111827
    );
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 35px;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

.stFileUploader {
    border-radius: 18px;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 10px;
}

.stButton>button {
    width: 100%;
    border: none;
    border-radius: 14px;
    height: 48px;
    font-size: 15px;
    font-weight: 600;
    color: white;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
}

.stDownloadButton>button {
    width: 100%;
    border: none;
    border-radius: 14px;
    height: 48px;
    font-size: 15px;
    font-weight: 600;
    color: white;
    background: linear-gradient(
        135deg,
        #059669,
        #10b981
    );
}

img {
    border-radius: 18px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "best_lane_detection_model.h5",
        compile=False
    )

    return model

# =========================================================
# MODEL LOADING
# =========================================================

try:
    model = load_model()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        Autonomous Lane Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Upload a road image to detect lane markings using Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MAIN CONTAINER
# =========================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")

        image = np.array(image)

        original = image.copy()

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        resized = cv2.resize(image, (256, 128))

        normalized = resized / 255.0

        input_image = np.expand_dims(
            normalized.astype(np.float32),
            axis=0
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with st.spinner("Detecting lanes..."):

            prediction = model.predict(
                input_image,
                verbose=0
            )

        # -------------------------------------------------
        # PROCESS MASK
        # -------------------------------------------------

        prediction = prediction[0]

        mask = prediction.squeeze()

        mask = (mask > 0.5).astype(np.uint8)

        # -------------------------------------------------
        # RESIZE MASK
        # -------------------------------------------------

        mask = cv2.resize(
            mask,
            (original.shape[1], original.shape[0])
        )

        # -------------------------------------------------
        # CREATE YELLOW LANE MASK
        # -------------------------------------------------

        lane_mask = np.zeros_like(original)

        # Bright Yellow
        lane_mask[:, :, 1] = 255
        lane_mask[:, :, 2] = 255

        # -------------------------------------------------
        # OVERLAY
        # -------------------------------------------------

        overlay = np.where(
            mask[:, :, np.newaxis] == 1,
            cv2.addWeighted(
                original,
                0.5,
                lane_mask,
                0.9,
                0
            ),
            original
        )

        overlay = overlay.astype(np.uint8)

        # -------------------------------------------------
        # DISPLAY IMAGES
        # -------------------------------------------------

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                original,
                caption="Original Image",
                use_container_width=True
            )

        with col2:

            st.image(
                overlay,
                caption="Lane Detection Result",
                use_container_width=True
            )

        # -------------------------------------------------
        # DOWNLOAD BUTTON
        # -------------------------------------------------

        result_image = Image.fromarray(overlay)

        buf = io.BytesIO()

        result_image.save(
            buf,
            format="PNG"
        )

        byte_im = buf.getvalue()

        st.write("")

        st.download_button(
            label="Download Result",
            data=byte_im,
            file_name="lane_detection_result.png",
            mime="image/png"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =========================================================
# CLOSE CARD
# =========================================================

st.markdown('</div>', unsafe_allow_html=True)