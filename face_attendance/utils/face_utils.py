import face_recognition
import cv2
import os
import pickle
import numpy as np
import time
from .db_utils import mark_attendance, get_batch_by_id

# In-memory queues for coordinating cross-request capture commands
CAPTURE_REQUESTS = {}  # e.g., {'enrollment_123': {'count': 0, 'done': False, 'batch_name': '...', 'subject': '...'}}
ATTENDANCE_STATE = {}

def get_encodings_path(batch_id):
    encodings_dir = "encodings"
    if not os.path.exists(encodings_dir):
        os.makedirs(encodings_dir)
    return os.path.join(encodings_dir, f"batch_{batch_id}_encodings.pkl")

def encode_batch_faces(batch_id, batch_name, subject):
    """
    Iterates through the dataset folder for a specific batch, generates encodings, 
    and saves them to a batch-specific pickle file.
    """
    known_encodings = {}
    dataset_path = os.path.join('dataset', batch_name, subject)
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset folder not found for batch {batch_name}.")
        return False

    print(f"[INFO] Encoding faces for batch {batch_name}... This might take a moment.")
    
    for enrollment_num in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, enrollment_num)
        if not os.path.isdir(user_folder):
            continue
            
        user_encodings = []
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            image = face_recognition.load_image_file(img_path)
            
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                user_encodings.append(encodings[0])
        
        if user_encodings:
            known_encodings[enrollment_num] = user_encodings
            print(f"[SUCCESS] Encoded user: {enrollment_num}")

    pkl_path = get_encodings_path(batch_id)
    with open(pkl_path, 'wb') as f:
        pickle.dump(known_encodings, f)
        
    print(f"[INFO] Encodings saved to {pkl_path}")
    return True

def generate_registration_frames(batch_name, subject):
    """
    Generator for web-based video streaming during registration.
    Continuously streams. If a capture request is active, it saves frames.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        faces_detected = len(face_locations)

        # Check if there is an active capture request
        active_req_key = None
        for key, req in CAPTURE_REQUESTS.items():
            if req.get('batch_name') == batch_name and req.get('subject') == subject and not req.get('done'):
                active_req_key = key
                break

        if active_req_key and faces_detected > 0:
            req = CAPTURE_REQUESTS[active_req_key]
            enrollment_number = req['enrollment']
            save_path = os.path.join('dataset', batch_name, subject, enrollment_number)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Use the first detected face for capture
            top, right, bottom, left = face_locations[0]
            face_img = frame[top:bottom, left:right]
            
            if face_img.size > 0 and req['count'] < 10:
                img_name = f"{enrollment_number}_{req['count']}.jpg"
                cv2.imwrite(os.path.join(save_path, img_name), face_img)
                req['count'] += 1
                
                if req['count'] >= 10:
                    req['done'] = True

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # UI Overlays
        if active_req_key:
            req = CAPTURE_REQUESTS[active_req_key]
            if not req['done']:
                cv2.putText(frame, f"Capturing: {req['count']}/10", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Capture Complete!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            status_text = "Face detected - Ready" if faces_detected > 0 else "Position face in frame"
            color = (0, 255, 0) if faces_detected > 0 else (0, 0, 255)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def generate_face_recognition_frames(batch_id, app=None):
    """
    Generator for web-based video streaming for attendance.
    """
    pkl_path = get_encodings_path(batch_id)
    encodings_by_user = {}
    
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            encodings_by_user = pickle.load(f)

    known_encodings = []
    known_labels = []
    for enrollment, enc_list in (encodings_by_user or {}).items():
        for enc in enc_list or []:
            known_encodings.append(np.asarray(enc))
            known_labels.append(enrollment)

    cap = cv2.VideoCapture(0)
    process_every_n = 2
    frame_index = 0
    recognized_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % process_every_n == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            new_recognized_faces = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    if any(matches):
                        distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_idx = int(np.argmin(distances))
                        if matches[best_idx]:
                            name = known_labels[best_idx]
                            if mark_attendance(batch_id, name) and app:
                                app.last_event = {"enrollment": name, "timestamp": time.time()}

                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                new_recognized_faces.append((top, right, bottom, left, name))
            recognized_faces = new_recognized_faces

        for (top, right, bottom, left, name) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, max(0, top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
