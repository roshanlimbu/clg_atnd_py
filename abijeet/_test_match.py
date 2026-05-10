"""
Quick diagnostic: capture one frame from camera and test FAISS matching.
Shows what similarity scores the internal team gets.
"""
import cv2
import numpy as np
from pathlib import Path
from database import DatabaseManager
from face_identity import FaceIdentityManager, EMBEDDING_DIM

DB_PATH = Path(__file__).resolve().parent / "attendance.db"
db = DatabaseManager(DB_PATH)
identity = FaceIdentityManager(db)

print(f"\n=== FAISS Index: {len(identity._known)} known faces, {identity._faiss_index.ntotal} vectors ===")
for kf in identity._known:
    print(f"  {kf.person_id:30s}  role={kf.role:10s}  embs={len(kf.embeddings)}  name={kf.display_name}")

# Capture one frame from webcam
print("\n--- Opening camera for 1 test frame ---")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

# Warm up camera (first few frames are often dark)
for _ in range(10):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: Could not capture frame")
    exit(1)

print(f"  Frame captured: {frame.shape}")

# Convert BGR to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Run InsightFace detection on full frame
faces = identity._model.get(frame)  # InsightFace expects BGR
print(f"  Faces detected in frame: {len(faces)}")

if not faces:
    print("  No faces found in camera frame. Make sure you're visible.")
    exit(1)

for i, face in enumerate(faces):
    emb = face.embedding
    if emb is None:
        print(f"  Face {i}: no embedding")
        continue
    
    emb = emb.ravel().astype(np.float32)
    print(f"\n  Face {i}: embedding dim={len(emb)}")
    
    # Check raw similarity against all known faces
    import faiss
    query = emb.reshape(1, -1).copy()
    faiss.normalize_L2(query)
    
    identity._ensure_index_current()
    D, I = identity._faiss_index.search(query, k=min(5, identity._faiss_index.ntotal))
    
    print(f"  Top matches (cosine similarity):")
    for j in range(len(I[0])):
        idx = int(I[0][j])
        sim = float(D[0][j])
        if idx == -1:
            continue
        # Find which person this index belongs to
        known_idx, emb_idx = identity._id_map[idx]
        kf = identity._known[known_idx]
        print(f"    {sim:.4f}  →  {kf.person_id} ({kf.display_name or 'no name'})")
    
    # Now test with identify()
    aligned = face.embedding  # Already have it
    # Use the crop for identify
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
    face_crop = rgb[y1:y2, x1:x2]
    
    if face_crop.size > 0:
        result = identity.identify(face_crop)
        print(f"\n  identify() result:")
        print(f"    person_id: {result.person_id}")
        print(f"    zone: {result.zone}")
        print(f"    raw_similarity: {result.raw_similarity:.4f}")
        print(f"    confidence: {result.confidence:.4f}")
        print(f"    role: {result.role}")
        print(f"    display_name: {result.display_name}")

print("\n=== Done ===")
