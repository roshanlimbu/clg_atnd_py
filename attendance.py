import cv2
import sqlite3
import numpy as np
import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

YOLO_MODEL_PATH = "yolov8n-face.pt"
COOLDOWN_MINUTES = 5

def cosine_similarity(a, b):
    # Calculate cosine similarity between two 1D vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_known_users():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM users")
    users = []
    for row in cursor.fetchall():
        user_id, name, emb_blob = row
        # Reconstruct numpy array from bytes
        emb = np.frombuffer(emb_blob, dtype=np.float32) 
        users.append({
            'id': user_id,
            'name': name,
            'embedding': emb
        })
    conn.close()
    return users

def log_attendance(user_id):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()

def get_face_embedding(app, face_crop):
    face_img = cv2.resize(face_crop, (112, 112))
    rec_model = app.models.get('recognition')
    if rec_model:
        emb = rec_model.get_feat(face_img)
        return np.array(emb).flatten()
    return None

def main():
    print("Loading users from database...")
    known_users = load_known_users()
    print(f"Loaded {len(known_users)} users.")

    print("Loading YOLO face model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("Loading InsightFace recognition model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    last_logged = {} # {user_id: datetime_object}

    cap = cv2.VideoCapture(0)
    print("Attendance tracking started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        display_frame = frame.copy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Expand box slightly
            h, w = frame.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - bw * 0.1))
            cy1 = max(0, int(y1 - bh * 0.1))
            cx2 = min(w, int(x2 + bw * 0.1))
            cy2 = min(h, int(y2 + bh * 0.1))

            face_crop = frame[cy1:cy2, cx1:cx2]
            
            label = "Unknown"
            color = (0, 0, 255) # Red for unknown
            
            if face_crop.size > 0:
                emb = get_face_embedding(app, face_crop)
                if emb is not None:
                    # Find best match
                    best_match = None
                    best_score = -1
                    
                    for user in known_users:
                        score = cosine_similarity(emb, user['embedding'])
                        if score > best_score:
                            best_score = score
                            best_match = user
                    
                    # Threshold for recognition (0.4 is a good starting point for cosine sim with ArcFace)
                    if best_score > 0.4: 
                        label = f"{best_match['name']} ({best_score:.2f})"
                        color = (0, 255, 0) # Green for recognized
                        
                        user_id = best_match['id']
                        now = datetime.datetime.now()
                        
                        # Check cooldown
                        if user_id not in last_logged or (now - last_logged[user_id]).total_seconds() > COOLDOWN_MINUTES * 60:
                            log_attendance(user_id)
                            last_logged[user_id] = now
                            print(f"Logged attendance for {best_match['name']} at {now.strftime('%H:%M:%S')}")
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Attendance System', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
