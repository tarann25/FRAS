import cv2
import face_recognition
import numpy as np
import sys

def validate_system():
    print(f"Python Version: {sys.version}")
    print("--------------------------------------------------")
    
    # 1. Check Camera
    print("[INFO] Checking webcam access...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Please check your camera settings.")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        return
    print("[SUCCESS] Webcam accessed successfully.")

    # 2. Check Face Recognition
    print("[INFO] Checking face_recognition library...")
    try:
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Attempt to detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"[INFO] Face locations found: {len(face_locations)}")
        
        if len(face_locations) > 0:
            # Attempt to encode
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if len(encodings) > 0:
                print(f"[SUCCESS] Generated encoding with shape: {encodings[0].shape}")
                if encodings[0].shape == (128,):
                    print("[PASS] System is ready for FRAS.")
                else:
                    print("[WARN] Encoding length is not 128 (unexpected).")
            else:
                print("[WARN] Detected face but could not encode.")
        else:
            print("[NOTE] No face detected in the single test frame. (This is fine if you weren't looking at the camera).")
            print("[PASS] Library loaded and executed detection function successfully.")
            
    except Exception as e:
        print(f"[ERROR] Face recognition check failed: {e}")

if __name__ == "__main__":
    validate_system()
