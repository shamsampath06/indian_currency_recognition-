# convert_model_old.py
from tensorflow.keras.models import load_model

model = load_model("currency_model.h5", compile=False)
model.save("currency_model.keras", save_format="keras")
print("✅ Model converted successfully!")
