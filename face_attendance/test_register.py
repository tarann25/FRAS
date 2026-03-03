from utils.face_utils import capture_user_images, encode_user_faces, recognize_faces_realtime
import os

def main():
    user_id = input("Enter User ID (e.g., your name): ").strip()
    if not user_id:
        print("[ERROR] User ID cannot be empty.")
        return

    # 1. Capture Images
    success = capture_user_images(user_id)
    
    if success:
        print(f"[SUCCESS] Images captured for {user_id}.")
        # 2. Encode
        encode_user_faces()
        #3. Recognize faces
        recognize_faces_realtime()
    else:
        print("[FAIL] Image capture interrupted or failed.")

if __name__ == "__main__":
    main()
