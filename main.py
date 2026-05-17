import cv2
import numpy as np
import os
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

actions = np.array(['left_jab', 'no_action']) 
model_path = 'action.h5'
threshold = 0.75 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose, lh, rh])

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

prev_time = time.perf_counter()
prev_dx, prev_dy = 0, 0
current_speed = 0.0

sequence = []
current_action = "..."
prob = 0.0

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        image, results = mediapipe_detection(frame, holistic)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        h, w, c = image.shape
        if results.pose_landmarks:
            left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
            dx, dy = left_wrist.x * w, left_wrist.y * h
            
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            if dt > 0:
                dist = np.sqrt((dx - prev_dx)**2 + (dy - prev_dy)**2)
                raw_speed = dist / dt
                current_speed = 0.7 * current_speed + 0.3 * raw_speed
            prev_time = curr_time
            prev_dx = dx
            prev_dy = dy
            
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] 
        
        if len(sequence) == 30:
            prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            action_idx = np.argmax(prediction)
            prob = prediction[action_idx]
            
            if prob > threshold: 
                current_action = actions[action_idx]
            else:
                current_action = "no_action"

        # UI Overlay
        cv2.rectangle(image, (0, 0), (w, 45), (245, 117, 16), -1)
        
        if current_action == 'left_jab':
            status_text = f"LEFT JAB ({prob*100:.1f}%)"
            text_color = (0, 255, 0)
        else:
            status_text = "NO ACTION"
            text_color = (255, 255, 255)

        cv2.putText(image, f"STATE: {status_text}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        
        cv2.putText(image, f"SPEED: {int(current_speed)} px/s", (w - 220, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Boxing Action Recognition", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()