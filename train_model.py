import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ✅ Ordered list of your categories
labels_ordered = ["Tennote", "Twentynote", "Fiftynote", "1Hundrednote", "2Hundrednote", "5Hundrednote", "2Thousandnote"]

# ✅ Path to dataset
data_dir = r"E:\indian_currency_recognition\dataset\Train"  # Change if needed

print(f"📁 Found folders in {data_dir} :", os.listdir(data_dir))

X = []
y = []

# ✅ Load and preprocess images
for label in labels_ordered:
    folder_path = os.path.join(data_dir, label)
    if not os.path.exists(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Could not read image: {img_path}")
            continue
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        X.append(img)
        y.append(labels_ordered.index(label))

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(labels_ordered))

print(f"✅ Total images loaded: {len(X)}")
print(f"✅ Shape of X: {X.shape}")
print(f"✅ Shape of y: {y.shape}")

# ✅ Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels_ordered), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
model.fit(X, y, epochs=10, batch_size=16)

# ✅ Save model
model.save("currency_model.h5")
print("✅ Model training complete and saved as currency_model.h5")
