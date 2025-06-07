'''Task-4
Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video
data, enabling intuitive human-computer interaction and gesture-based control systems.'''
import os
import shutil
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

RAW_DATASET_DIR = r'D:\task\task4\Images of American Sign Language (ASL) Alphabet Gestures\Images of American Sign Language (ASL) Alphabet Gestures\Root\Root\Type_01_(Raw_Gesture)'
BASE_DIR = r'D:\task\task4\dataset\processed'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
SPLIT_RATIO = 0.8  # 80% train, 20% val

def prepare_dataset():
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print("✅ Dataset already prepared. Skipping split.")
        return

    print("📁 Creating train/val folders...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    for class_name in os.listdir(RAW_DATASET_DIR):
        class_path = os.path.join(RAW_DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_name, img))

        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, class_name, img))

    print("✅ Dataset split complete.")

prepare_dataset()

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()


model.fit(train_data, validation_data=val_data, epochs=EPOCHS)


model.save('hand_gesture_model.h5')
print("✅ Model saved as 'hand_gesture_model.h5'.")

def predict_from_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=0)
    pred = model.predict(reshaped)
    label = class_names[np.argmax(pred)]
    return label

"""
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    label = predict_from_frame(frame)
    cv2.putText(frame, f'Prediction: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
print("👋 Gesture recognition model trained and saved.")
