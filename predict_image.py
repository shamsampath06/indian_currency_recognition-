import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load model
model = load_model('currency_model.h5')

# ✅ Image path
image_path = r"E:\indian_currency_recognition\IMG_001.jpg"

# Label map
label_map = {
    10: "Tennote",
    20: "Twentynote",
    50: "Fiftynote",
    100: "1Hundrednote",
    200: "2Hundrednote",
    500: "5Hundrednote",
    2000: "2Thousandnote"
}

# Reverse label map
reverse_label_map = {v: k for k, v in label_map.items()}

# Ordered list as per model output
labels_ordered = ["Tennote", "Twentynote", "Fiftynote", "1Hundrednote", "2Hundrednote", "5Hundrednote", "2Thousandnote"]

# Load and preprocess image
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Could not read image: {image_path}")
    exit()

img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_label = labels_ordered[predicted_index]
predicted_value = reverse_label_map[predicted_label]

# Output
print("🧾 Predicted Currency Note:", predicted_label)
print("💰 Value: ₹", predicted_value)
