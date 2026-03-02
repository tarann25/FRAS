# Face recognition utilities
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
