import argparse
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


# Define model paths
Face_model = "backend/models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
Face_text = "backend/models/face_detection/deploy.prototxt.txt"
Age_model = "backend/models/age_gender_detection/age_net.caffemodel"
Age_text = "backend/models/age_gender_detection/age_deploy.prototxt"
Gender_model = "backend/models/age_gender_detection/gender_net.caffemodel"
Gender_text = "backend/models/age_gender_detection/gender_deploy.prototxt"


# Verify files exist
for file_path in [Face_model, Face_text, Age_model, Age_text, Gender_model, Gender_text]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")


# Define lists for age and gender labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def detect_face(frame, net):


    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (1000, 1000)), 1.0,
                                 (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces



def detect_age(face_roi, age_net):


    face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
    age_net.setInput(face_blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return age


def detect_gender(face_roi, gender_net):

    gender_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
    gender_net.setInput(gender_blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    return gender


def main():


    # Load face detection model
    face_net = cv2.dnn.readNetFromCaffe(Face_text, Face_model)

    # Load age and gender prediction models
    print("loading age and gender models...")
    age_net = cv2.dnn.readNetFromCaffe(Age_text, Age_model)
    gender_net = cv2.dnn.readNetFromCaffe(Gender_text, Gender_model)

    # Initialize the video stream
    print("starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # Loop over the frames from the video stream
    while True:
        # Grab the frame and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # Detect faces in the frame
        faces = detect_face(frame, face_net)

        for (startX, startY, endX, endY) in faces:
            face_roi = frame[startY:endY, startX:endX]

            # Predict age
            age = detect_age(face_roi, age_net)

            # Predict gender
            gender = detect_gender(face_roi, gender_net)

            # Draw bounding box and display results
            text = f"{gender}, {age}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (255,192,203), 1)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)

        # Show the output frame
        cv2.imshow("Video Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if key == ord("q"):
            break

    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()