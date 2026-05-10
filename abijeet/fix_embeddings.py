"""
Re-encode internal team members that have old 128-d dlib embeddings.
Reads their reference photos and recomputes ArcFace 512-d embeddings.
"""
import sys
from pathlib import Path
from database import DatabaseManager
from face_identity import FaceIdentityManager, EMBEDDING_DIM
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "attendance.db"

db = DatabaseManager(DB_PATH)
identity = FaceIdentityManager(db)

# Get all internal team members
internal = db.get_internal_persons()
print(f"\n=== Internal Team Members: {len(internal)} ===")

fixed = 0
for member in internal:
    pid = member["person_id"]
    name = member["display_name"]
    photo = member["reference_photo_path"]

    # Check if their embedding is old format
    person = db.get_person(pid)
    if person is None:
        print(f"  SKIP {pid}: not found in persons table")
        continue

    sig = person["face_signature"]
    embeddings = identity._deserialize(sig)

    if embeddings and len(embeddings[0]) == EMBEDDING_DIM:
        print(f"  OK   {pid}: already has {EMBEDDING_DIM}-d ArcFace embedding ({name})")
        continue

    # Need to recompute
    print(f"  FIX  {pid}: has {len(embeddings[0]) if embeddings else 0}-d embedding, needs recompute ({name})")

    if not photo:
        print(f"       ERROR: no reference photo path for {pid}")
        continue

    photo_path = PROJECT_DIR / photo
    if not photo_path.is_file():
        print(f"       ERROR: photo not found at {photo_path}")
        continue

    # Load image and compute new ArcFace embedding
    try:
        image = Image.open(str(photo_path)).convert("RGB")
        rgb = np.array(image)
        new_sig = identity.compute_signature(rgb)

        if new_sig is None:
            print(f"       ERROR: ArcFace could not detect a face in {photo_path.name}")
            continue

        # Update the database
        ok = db.update_person_signature(pid, new_sig)
        if ok:
            print(f"       SUCCESS: updated {pid} with new {EMBEDDING_DIM}-d embedding")
            fixed += 1
        else:
            print(f"       ERROR: database update failed for {pid}")
    except Exception as e:
        print(f"       ERROR: {e}")

print(f"\n=== Fixed {fixed} embeddings ===")
if fixed > 0:
    print("Restart main.py for changes to take effect (or wait 30 seconds for auto-refresh).")
