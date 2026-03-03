from utils.face_utils import capture_user_images, encode_user_faces, generate_face_recognition_frames
from utils.db_utils import init_db, add_user
import os
import cv2
import numpy as np

def main():
    # 0. Initialize Database
    init_db()

    user_id = input("Enter User ID (e.g., your name): ").strip()
    if not user_id:
        print("[ERROR] User ID cannot be empty.")
        return

    # 1. Capture Images
    success = capture_user_images(user_id)
    
    if success:
        print(f"[SUCCESS] Images captured for {user_id}.")
        # 1b. Add user to database
        user_name = input(f"Enter display name for {user_id}: ").strip() or user_id
        add_user(user_id, user_name)

        # 2. Encode
        encode_user_faces()

        # 3. Ask for subject
        subject = input("Enter Subject Name (e.g., Math, Physics): ").strip() or "General"

        # 4. Recognize faces (CLI version)
        print("[INFO] Starting realtime recognition. Press ENTER or 'q' to quit.")
        for frame_bytes_with_header in generate_face_recognition_frames(subject):
            # Extract the raw jpeg bytes from the multipart stream
            # The generator yields: b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            # We just need to find the jpeg start and end or just decode carefully.
            # For simplicity, we'll just parse out the bytes after the \r\n\r\n
            try:
                parts = frame_bytes_with_header.split(b'\r\n\r\n')
                if len(parts) > 1:
                    jpeg_bytes = parts[1].split(b'\r\n')[0]
                    # Convert to numpy array
                    nparr = np.frombuffer(jpeg_bytes, np.uint8)
                    # Decode to image
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        cv2.imshow("Face Recognition (CLI) - Press ENTER or 'q' to quit", frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 13:
                            break
            except Exception as e:
                print(f"[ERROR] Stream decoding error: {e}")
                break
        
        cv2.destroyAllWindows()
    else:
        print("[FAIL] Image capture interrupted or failed.")

if __name__ == "__main__":
    main()
