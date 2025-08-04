import streamlit as st
import cv2
import numpy as np
from tensorflow import keras  # ✅ use tf.keras correctly

# Load the converted model
try:
    model = keras.models.load_model("currency_model.keras", compile=False)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Define labels & values
labels_ordered = [
    "Tennote", "Twentynote", "Fiftynote", "1Hundrednote",
    "2Hundrednote", "5Hundrednote", "2Thousandnote"
]
values = [10, 20, 50, 100, 200, 500, 2000]

# Page config
st.set_page_config(page_title="Indian Currency Recognition", page_icon="💵", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f7, #d9eaf7);
    }
    .title {
        font-size: 2.3em; 
        font-weight: bold; 
        color: #003366;
    }
    .stSidebar > div:first-child {
        background: #003366;
        color: white;
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Logo & title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.jpeg", width=80)
with col2:
    st.markdown('<div class="title">💵 Indian Currency Recognition</div>', unsafe_allow_html=True)

st.write("Upload an image or capture from webcam to detect the note denomination.")

# Sidebar
with st.sidebar:
    st.image("logo.jpeg", width=120)
    st.title("⚙️ Options")
    st.info("Developed by **Your Name**")

# Upload or camera input
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or capture photo")

# Prediction function
def predict(img):
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    return labels_ordered[index], values[index], confidence

# Handle input
image = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Show prediction
if image is not None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="🖼️ Your Image", use_column_width=True)
    with st.spinner("🔍 Predicting..."):
        label, value, confidence = predict(image)
    st.success(f"🧾 Predicted Note: **{label}**")
    st.info(f"💰 Value: ₹ **{value}**")
    st.write(f"✅ Confidence: **{confidence:.2f}%**")

st.markdown("---")
st.caption("✨ Built with TensorFlow, OpenCV & Streamlit")
