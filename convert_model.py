from tensorflow import keras

# Load the old .h5 model
model = keras.models.load_model("currency_model.h5", compile=False)

# Save in new .keras format
model.save("currency_model.keras", save_format="keras_v3")

print("✅ Conversion successful! New file: currency_model.keras")
