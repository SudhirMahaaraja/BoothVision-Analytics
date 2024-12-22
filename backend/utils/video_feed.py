import cv2
import time
from face_utils import detect_face_and_predict
from database_handler import connect_db, save_visitor_details


def annotate_frame(frame, face, gender, age):
    """
    Annotate the frame with detection results.

    Args:
        frame: Video frame to annotate.
        face: Coordinates of the detected face.
        gender: Detected gender.
        age: Detected age group.
    """
    (x, y, w, h) = face
    label = f"{gender}, {age}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def process_video_feed():
    """
    Process the video feed, perform face detection, gender and age estimation,
    and save the results to the database.
    """
    # Initialize the database
    connect_db()
    print("Database connected successfully.")

    # Start capturing video feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Video feed started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Perform face, age, and gender detection
        faces, genders, ages = detect_face_and_predict(frame)

        if len(faces) == 0:
            print("No faces detected.")
        else:
            for i, face in enumerate(faces):
                gender = genders[i]
                age = ages[i]
                person_id = f"person_{time.time()}"  # Unique ID based on timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp

                # Save detection data to the database
                save_visitor_details(person_id, timestamp, gender, age)

                # Annotate the frame with the detection results
                annotate_frame(frame, face, gender, age)

        # Display the annotated frame
        cv2.imshow("Booth Analytics - Video Feed", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video feed.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        print("Loading models...")
        print("Models loaded successfully.")
        process_video_feed()
    except Exception as e:
        print(f"An error occurred: {e}")
