import cv2
import dlib
from backend.utils.database_handler import save_visitor_details
import numpy as np
import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_PROTO = os.path.join(BASE_DIR, "../models/face_detection/opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(BASE_DIR, "../models/face_detection/opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(BASE_DIR, "../models/age_gender_detection/age_deploy.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "../models/age_gender_detection/age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "../models/age_gender_detection/gender_deploy.prototxt")
GENDER_MODEL = os.path.join(BASE_DIR, "../models/age_gender_detection/gender_net.caffemodel")

# Verify files exist
for file_path in [FACE_MODEL, FACE_PROTO, AGE_MODEL, AGE_PROTO, GENDER_MODEL, GENDER_PROTO]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

# Load models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

print("Models loaded successfully.")

# Age and Gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']


# Generate unique ID for each face (can be enhanced for real-world applications)
def generate_unique_id(frame, face_id):
    return f"{face_id}_{str(frame)}"


# Perform face detection and gender/age prediction
def detect_face_and_predict(frame):
    h, w = frame.shape[:2]

    # Convert frame to blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only consider face with confidence > 0.5
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Crop face from frame
            face = frame[startY:endY, startX:endX]

            # Predict age and gender
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4, 87.5, 114.0), swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()

            # Get age and gender predictions
            age = age_list[np.argmax(age_preds)]
            gender = gender_list[np.argmax(gender_preds)]

            # Generate a unique ID for this face (based on frame and face)
            unique_id = generate_unique_id(frame, i)

            # Save to database
            save_visitor_details(unique_id, gender, age, "Neutral")  # Emotion can be enhanced later

            # Draw rectangle and add labels
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
