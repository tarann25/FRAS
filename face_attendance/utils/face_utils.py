# Face recognition utilities
from typing import Any
from numpy._typing._array_like import NDArray
import face_recognition
import cv2
import os
import pickle
import numpy as np

def capture_user_images(user_id, num_samples=10):
    """
    Captures samples of a user's face from the webcam and saves them.
    """
    save_path = os.path.join('dataset', user_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"[INFO] Starting capture for user: {user_id}. Look at the camera...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Detect face for visual feedback and cropping
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            # Draw rectangle on the live preview
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Save the cropped face image
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                img_name = f"{user_id}_{count}.jpg"
                cv2.imwrite(os.path.join(save_path, img_name), face_img)
                count += 1
                print(f"[INFO] Captured image {count}/{num_samples}")

        cv2.imshow("Registering User - Press 'q' to quit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return count == num_samples

def encode_user_faces():
    """
    Iterates through the dataset folder, generates encodings, and saves them to a pickle file.
    """
    known_encodings = {}
    dataset_path = 'dataset'
    
    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset folder not found.")
        return False

    print("[INFO] Encoding faces... This might take a moment.")
    
    for user_id in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user_id)
        if not os.path.isdir(user_folder):
            continue
            
        user_encodings = []
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            image = face_recognition.load_image_file(img_path)
            
            # Generate encoding (taking the first one found in the image)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                user_encodings.append(encodings[0])
        
        if user_encodings:
            # We store the average encoding for better stability or just a list
            # For simplicity in this workflow, we'll store the list of encodings per user
            known_encodings[user_id] = user_encodings
            print(f"[SUCCESS] Encoded user: {user_id}")

    # Save to pickle
    encodings_dir = 'encodings'
    if not os.path.exists(encodings_dir):
        os.makedirs(encodings_dir)
        
    with open(os.path.join(encodings_dir, 'encodings.pkl'), 'wb') as f:
        pickle.dump(known_encodings, f)
        
    print(f"[INFO] Encodings saved to {os.path.join(encodings_dir, 'encodings.pkl')}")
    return True

def recognize_faces_realtime():
    """
    Runs webcam-based face recognition using saved encodings.

    Looks for `encodings/encodings.pkg` first, then falls back to `encodings/encodings.pkl`.
    The file is expected to contain a dict like: {user_id: [encoding1, encoding2, ...], ...}.
    """
    encodings_dir = "encodings"
    pkg_path = os.path.join(encodings_dir, "encodings.pkg")
    pkl_path = os.path.join(encodings_dir, "encodings.pkl")

    enc_path = pkg_path if os.path.exists(pkg_path) else pkl_path
    if not os.path.exists(enc_path):
        print(f"[ERROR] Encodings file not found. Expected `{pkg_path}` or `{pkl_path}`.")
        return False

    with open(enc_path, "rb") as f:
        encodings_by_user = pickle.load(f)

    known_encodings = []
    known_labels = []
    for user_id, enc_list in (encodings_by_user or {}).items():
        for enc in enc_list or []:
            known_encodings.append(np.asarray(enc))
            known_labels.append(user_id)

    if not known_encodings:
        print(f"[ERROR] No encodings found inside `{enc_path}`.")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return False

    print("[INFO] Starting realtime recognition. Press 'q' to quit.")

    process_every_n = 2 #runs every 2 frames
    frame_index = 0
    recognized_faces = []  # list of (top, right, bottom, left, name) on last processed frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame_index += 1

            # Run heavy recognition only every 2 frames
            if frame_index % process_every_n == 0:
                # Resize frame to 1/4 size for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Use the faster HOG model for CPU-based detection
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                new_recognized_faces = []

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"

                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    if any(matches):
                        distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_idx = int(np.argmin(distances))
                        if matches[best_idx]:
                            name = known_labels[best_idx]

                    # Scale face locations back up since we detected on a 1/4 size frame
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    new_recognized_faces.append((top, right, bottom, left, name))

                recognized_faces = new_recognized_faces

            # Draw the most recently recognized faces on the full-size frame
            for (top, right, bottom, left, name) in recognized_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (left, max(0, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True