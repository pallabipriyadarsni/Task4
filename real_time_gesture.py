import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('hand_gesture_model.h5')
class_names = ['A', 'B', 'C', 'D','DELL', 'E', 'F', 'G', 'H', 'I', 'J',
'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','SPACE','T',
'U', 'V', 'W', 'X', 'Y', 'Z'
]  

IMG_SIZE = 64

cap = cv2.VideoCapture(0)
print("Starting real-time hand gesture recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    if 0 <= predicted_class < len(class_names):
        gesture_name = class_names[predicted_class]
    else:
        gesture_name = "Unknown"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'Gesture: {gesture_name}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()