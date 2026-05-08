import os
import cv2
import sqlite3
import numpy as np
import urllib.request
from ultralytics import YOLO
from insightface.app import FaceAnalysis

YOLO_MODEL_PATH = "yolov8n-face.pt"
YOLO_MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"

def download_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Downloading YOLO face model from {YOLO_MODEL_URL}...")
        urllib.request.urlretrieve(YOLO_MODEL_URL, YOLO_MODEL_PATH)
        print("Download complete.")

def get_face_embedding(app, face_crop):
    """
    Given a FaceAnalysis app and a cropped face image (numpy array),
    return the 512D embedding.
    """
    face_img = cv2.resize(face_crop, (112, 112))
    rec_model = app.models.get('recognition')
    if rec_model:
        emb = rec_model.get_feat(face_img)
        # flatten in case it returns an array of shape (1, 512)
        return np.array(emb).flatten()
    return None

def main():
    name = input("Enter user name to register: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    download_yolo_model()

    print("Loading YOLO face model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("Loading InsightFace recognition model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    print("Camera opened. Press 'c' to capture face, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Detect faces with YOLO
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        display_frame = frame.copy()

        # Just draw boxes for preview
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Register User', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cancelled.")
            break
        elif key == ord('c'):
            if len(boxes) == 0:
                print("No face detected! Try again.")
            elif len(boxes) > 1:
                print("Multiple faces detected! Please ensure only one face is in the frame.")
            else:
                x1, y1, x2, y2 = map(int, boxes[0][:4])
                
                # Expand bounding box slightly (10%) to ensure whole face is included
                h, w = frame.shape[:2]
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - bw * 0.1))
                y1 = max(0, int(y1 - bh * 0.1))
                x2 = min(w, int(x2 + bw * 0.1))
                y2 = min(h, int(y2 + bh * 0.1))

                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    print("Invalid crop. Try again.")
                    continue

                print("Extracting embedding...")
                emb = get_face_embedding(app, face_crop)
                
                if emb is not None:
                    # Save to SQLite DB
                    conn = sqlite3.connect('attendance.db')
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, emb.tobytes()))
                    conn.commit()
                    conn.close()
                    print(f"User '{name}' registered successfully!")
                    break
                else:
                    print("Failed to extract embedding.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
