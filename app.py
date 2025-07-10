import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("currency_model.h5")

# Define labels (must match training order)
labels_ordered = ["Tennote", "Twentynote", "Fiftynote", "1Hundrednote", "2Hundrednote", "5Hundrednote", "2Thousandnote"]
values = [10, 20, 50, 100, 200, 500, 2000]

st.title("💵 Indian Currency Note Recognition")
st.write("Upload an image or capture from your webcam to predict the currency note.")

# 📷 Live camera input
camera_image = st.camera_input("Take a picture")

# 📁 Or file upload
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

def predict_image(img):
    # Preprocess
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # Predict
    prediction = model.predict(img_resized)
    predicted_index = np.argmax(prediction)
    predicted_label = labels_ordered[predicted_index]
    predicted_value = values[predicted_index]
    return predicted_label, predicted_value

# Handle camera capture
if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Captured Image", use_column_width=True)
        label, value = predict_image(img)
        st.success(f"🧾 Predicted Currency Note: {label}")
        st.info(f"💰 Value: ₹ {value}")
    else:
        st.error("❌ Could not read the captured image.")

# Handle file upload
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Uploaded Image", use_column_width=True)
        label, value = predict_image(img)
        st.success(f"🧾 Predicted Currency Note: {label}")
        st.info(f"💰 Value: ₹ {value}")
    else:
        st.error("❌ Could not read the uploaded image.")
