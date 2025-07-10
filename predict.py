# predict.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

model = load_model("currency_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)

def predict_currency(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    label = label_map[str(class_idx)]
    confidence = prediction[0][class_idx] * 100
    return label, confidence
