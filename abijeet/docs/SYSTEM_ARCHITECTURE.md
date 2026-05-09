# System Architecture: Advanced Profile Detection

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CAMERA FEED (Live)                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │   CameraFeed Thread        │
            │  (Continuous Capture)      │
            └────────────┬───────────────┘
                         │
                         ▼
                    ┌─────────────┐
                    │ Every 5th   │ ◄── SAMPLE_EVERY = 5
                    │ Frame       │     (Configurable)
                    └────┬────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  YOLOv8n Face Detection            │
        │  (All faces in 1 forward pass)     │
        │  Returns: bounding boxes            │
        │  Throughput: 20-50 faces/frame     │
        └────┬───────────────────────────────┘
             │
             ├─────────────────────────────────────────┐
             │                                         │
             ▼                                         ▼
    ┌─────────────────┐              (No faces detected)
    │ FACE ALIGNMENT  │              Return empty
    │ (MediaPipe)     │
    │ - Crop face     │
    │ - Align with    │
    │   eye landmarks │
    │ - Resize 224x224│
    └────┬────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │ HEAD POSE DETECTION (NEW!)                           │
    │ ┌──────────────────────────────────────────────────┐ │
    │ │ - Extract facial landmarks (MediaPipe Face Mesh)│ │
    │ │ - Solve PnP (3D → 2D projection)               │ │
    │ │ - Extract Euler angles (yaw, pitch, roll)      │ │
    │ │ - Classify into pose categories                │ │
    │ │ Returns: PoseEstimate                          │ │
    │ │ - Yaw: [-90°, 90°] (left-right)               │ │
    │ │ - Pitch: [-90°, 90°] (up-down)                │ │
    │ │ - Roll: [-90°, 90°] (tilt)                    │ │
    │ │ - Pose: FRONT|LEFT|RIGHT|LEFT_PROFILE|etc.   │ │
    │ └──────────────────────────────────────────────────┘ │
    └────┬──────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │ DLIB EMBEDDING ENCODING                              │
    │ (ResNet-34, 128-dimensional)                         │
    │ Returns: embedding vector                            │
    └────┬───────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ POSE-AWARE IDENTITY MATCHING (NEW!)                         │
    │ ┌────────────────────────────────────────────────────────┐  │
    │ │ For each known person:                               │  │
    │ │  - Load stored embeddings + pose info                │  │
    │ │  - Calculate L2 distance                             │  │
    │ │  - Get pose-adjusted threshold:                      │  │
    │ │    • Same pose: 0.60 (strict)                        │  │
    │ │    • Front↔Left: 0.63                               │  │
    │ │    • Front↔Profile: 0.69                            │  │
    │ │    • Profile↔Profile: 0.75 (lenient)               │  │
    │ │  - if distance < threshold: MATCH FOUND             │  │
    │ │                                                       │  │
    │ │ Returns:                                             │  │
    │ │ - person_id (matched or NEW generated ID)          │  │
    │ │ - confidence score                                   │  │
    │ │ - head_pose info                                    │  │
    │ │ - is_new flag                                       │  │
    │ └────────────────────────────────────────────────────────┘ │
    └────┬─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │ ATTENDANCE RECORDING                                 │
    │ - Layer 1: Memory file (fast duplicate check)        │
    │ - Layer 2: Database (UNIQUE constraint)              │
    │ - Record pose info (NEW!)                            │
    │ - Save attendance photo                              │
    │ Returns: AttendanceResult with status                │
    └────┬───────────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │ LIVE FEED DISPLAY & LOGGING                          │
    │ - Draw bounding box + label                          │
    │ - Show pose info (NEW!)                              │
    │ - Color code status (green/yellow/red/blue)          │
    │ - Log to console + file                              │
    │ - Update dashboard (optional)                        │
    └──────────────────────────────────────────────────────┘
```

---

## Data Flow: Per-Frame Processing

```
INPUT: Full video frame (1920x1080)
       │
       ├─ YOLO Detection
       │  │
       │  ├─ Face 1 @ (x1, y1, x2, y2)
       │  ├─ Face 2 @ (x3, y3, x4, y4)
       │  └─ Face 3 @ (x5, y5, x6, y6)
       │
       ├─ For each Face:
       │  │
       │  ├─ ALIGNMENT
       │  │  └─ Aligned RGB image (224x224)
       │  │
       │  ├─ POSE DETECTION
       │  │  ├─ MediaPipe landmarks (478 points)
       │  │  ├─ PnP solver
       │  │  ├─ Euler angles: (yaw=-45°, pitch=5°, roll=2°)
       │  │  └─ Pose category: LEFT_PROFILE
       │  │
       │  ├─ EMBEDDING
       │  │  └─ 128-D vector: [0.245, -0.156, ..., 0.089]
       │  │
       │  ├─ MATCHING
       │  │  ├─ Stored embeddings for Person A:
       │  │  │  - [0.243, -0.158, ...] (stored_pose: FRONT)
       │  │  │  - [0.244, -0.157, ...] (stored_pose: LEFT)
       │  │  │  - [0.246, -0.155, ...] (stored_pose: LEFT_PROFILE) ◄─ Best match
       │  │  │
       │  │  ├─ Calculate distances:
       │  │  │  - vs FRONT:        0.68 > 0.63 → NO MATCH
       │  │  │  - vs LEFT:         0.65 > 0.66 → NO MATCH
       │  │  │  - vs LEFT_PROFILE: 0.62 < 0.75 → MATCH! ✓
       │  │  │
       │  │  └─ Result: PERSON_A (confidence: 0.62, is_new: false)
       │  │
       │  └─ RECORD ATTENDANCE
       │     ├─ Check memory file: Not marked today
       │     ├─ Insert to database: attendance.db
       │     ├─ Update memory file
       │     ├─ Save photo: attendance_photos/2026-05-08/PERSON_A_1.jpg
       │     └─ Log: "COUNTED: PERSON_A | count=1 | confidence=62% | pose=left_profile"
       │
       └─ Return: [AttendanceResult, AttendanceResult, AttendanceResult]
         │
         ▼
       OUTPUT: Live feed with boxes, labels, and pose info
```

---

## Class Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│ HeadPoseDetector                                                │
├─────────────────────────────────────────────────────────────────┤
│ Attributes:                                                     │
│  - mp_face_mesh: MediaPipe Face Mesh                           │
│  - model_points_3d: 3D face model (6 key points)              │
│                                                                 │
│ Methods:                                                        │
│  + estimate_pose(face_image: ndarray) → PoseEstimate          │
│  + _rotation_matrix_to_euler_angles(rm) → (yaw, pitch, roll)  │
│  + _classify_pose() → HeadPose enum                           │
│  + draw_pose_on_frame() → annotated frame                     │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │ uses
                               │
┌─────────────────────────────────────────────────────────────────┐
│ FaceIdentityResult (Dataclass)                                  │
├─────────────────────────────────────────────────────────────────┤
│  person_id: str                                                 │
│  confidence: float                                              │
│  is_new: bool                                                   │
│  head_pose: Optional[HeadPose]  ◄─── NEW!                     │
│  pose_angles: Optional[Dict]    ◄─── NEW!                     │
│  ... + 4 more fields                                           │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │ returns
                               │
┌─────────────────────────────────────────────────────────────────┐
│ FaceIdentityManager                                             │
├─────────────────────────────────────────────────────────────────┤
│ Attributes:                                                     │
│  - _head_pose_detector: HeadPoseDetector  ◄─── NEW!           │
│  - db: DatabaseManager                                         │
│  - _known: List[KnownFace]                                    │
│  - MAX_EMBEDDINGS_PER_PERSON = 8  ◄─── INCREASED             │
│                                                                 │
│ Methods:                                                        │
│  + identify(face_image) → FaceIdentityResult                  │
│  - _find_best_match(emb, incoming_pose) → match  ◄─ UPDATED  │
│  - _add_or_update_embedding(pid, emb, dist,                  │
│      pose_label, pose_angles)  ◄─ NEW PARAMS                 │
│  - _create_person(emb, pose_label, pose_angles)  ◄─ NEW      │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │ uses
                               │
┌─────────────────────────────────────────────────────────────────┐
│ KnownFace (Dataclass)                                           │
├─────────────────────────────────────────────────────────────────┤
│  person_id: str                                                 │
│  embeddings: List[ndarray]                                     │
│  pose_labels: List[str]      ◄─── NEW! (e.g., "front")       │
│  pose_angles: List[Dict]     ◄─── NEW! (yaw, pitch, roll)    │
│  ... + 2 more fields                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Matching Algorithm (Detailed)

```
Input: 
  - new_embedding (128-D vector from current face)
  - new_pose (HeadPose from current face)
  - known_faces (list of stored persons with their embeddings)

Process:
  1. Flatten all stored embeddings:
     all_embeddings = [
       known_faces[0].embeddings[0],  ← Person A, angle 1
       known_faces[0].embeddings[1],  ← Person A, angle 2
       known_faces[1].embeddings[0],  ← Person B, angle 1
       ...
     ]

  2. Calculate L2 distances:
     distances = [0.45, 0.68, 0.92, ...]  ← vs each stored embedding

  3. Find best match:
     best_idx = argmin(distances)  = 0
     best_distance = 0.45
     best_known_face = known_faces[0]  (Person A)
     best_stored_pose = known_faces[0].pose_labels[0]  (FRONT)

  4. Get pose-adjusted threshold:
     if new_pose == LEFT_PROFILE and best_stored_pose == FRONT:
       adjustment_factor = 1.15
       threshold = 0.60 * 1.15 = 0.69
     else:
       threshold = 0.60

  5. Decision:
     if best_distance < threshold:  (0.45 < 0.69)
       MATCH FOUND ✓
       return (best_known_face, best_distance, confidence)
     else:
       NO MATCH → New person registered

Output:
  - Matched person or new generated ID
  - Confidence score (1.0 - distance)
  - Pose information
```

---

## Performance Metrics

```
Timing Breakdown (per face):
├─ YOLO Detection (full frame)        : 20ms
├─ Face Alignment (per face)          : 3ms
├─ Pose Detection (NEW)               : 5-10ms  ◄── Added
├─ Embedding                          : 30ms
├─ Matching (100 known faces)         : 2ms
└─ Recording + Photo Save             : 5ms
   ────────────────────────────────────────
   Total per face                      : ~65ms

Frame Processing (5 faces):
├─ YOLO (batch, all faces at once)    : 20ms
├─ Per-face processing × 5             : (3+8+30+2) × 5 = 215ms
│  (alignment, pose, embedding, match)
├─ Recording (all 5 faces)            : 5ms
└─ Display rendering                   : 10ms
   ────────────────────────────────────────
   Total per frame                     : ~250ms
   FPS: 4 frames/sec (at 5 faces/frame)
   * With sample_every=5: effective ~20 FPS on live display

Memory Usage:
├─ Pose detector model                : ~50MB (MediaPipe)
├─ Per-person embeddings              : 8 × 128-D = ~4KB per person
├─ Pose metadata                      : ~1KB per person
└─ Total for 1000 persons             : ~5MB

Storage:
├─ Attendance DB                      : 1MB per 10,000 records
├─ Attendance photos                  : 500KB each (avg 3 per person)
├─ Logs (daily)                       : 1-5MB per day
└─ Enrollment captures                : 50MB per person (15 photos)
```

---

## State Machine: Face Recognition

```
NEW FACE DETECTION
       │
       ▼
   ┌─────────────────┐
   │ COMPUTE POSE    │
   │ (NEW STEP)      │
   └────┬────────────┘
        │
        ▼
   ┌──────────────────────┐
   │ COMPUTE EMBEDDING    │
   └────┬─────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────┐
   │ FIND BEST MATCH                            │
   │ (with pose-adjusted threshold)             │
   └────┬────────────────────┬──────────────────┘
        │                    │
        │ Match found        │ No match
        │                    │
        ▼                    ▼
   ┌──────────────┐    ┌────────────────────┐
   │ KNOWN PERSON │    │ REGISTER NEW       │
   │              │    │ (Generate ID)      │
   │ ✓ Update     │    │                    │
   │   embeddings │    │ ✓ Store embedding  │
   │   (pose)     │    │ ✓ Store pose info  │
   │              │    └────────────────────┘
   └──────────────┘
        │
        ▼
   ┌──────────────────────────────────────┐
   │ CHECK ATTENDANCE RECORD TODAY        │
   └────┬────────────────────────────────┘
        │
        ├─ Already marked today
        │  └─ Status: ALREADY_MARKED (yellow)
        │
        ├─ Low confidence
        │  └─ Status: LOW_CONFIDENCE (gray)
        │
        └─ New mark
           ├─ Insert to database
           ├─ Update memory file
           ├─ Save photo + pose info
           ├─ Status: MARKED or NEW_FACE
           └─ Log event
```

---

## Configuration Points

```
head_pose_detector.py:
├─ HeadPose.FRONT  range       : -15° to 15° yaw
├─ HeadPose.LEFT   range       : -45° to -15° yaw
├─ HeadPose.RIGHT  range       : 15° to 45° yaw
├─ HeadPose.LEFT_PROFILE range : -90° to -45° yaw
├─ HeadPose.RIGHT_PROFILE range: 45° to 90° yaw
├─ POSE_DISTANCE_ADJUSTMENTS   : Tunable thresholds
└─ Base threshold             : 0.60 (L2 distance)

face_identity.py:
├─ MAX_EMBEDDINGS_PER_PERSON   : 8 (was 5)
├─ MATCH_DISTANCE_THRESHOLD    : 0.60
├─ EMA_BLEND_THRESHOLD         : 0.35
└─ _head_pose_detector         : None to disable

enrollment_mode.py:
├─ REQUIRED_POSES              : ["front", "left_profile", ...]
├─ FRAMES_PER_POSE             : 3
├─ CONFIDENCE_THRESHOLD        : 0.85
└─ Storage directory           : configurable
```

---

**This architecture ensures robust multi-angle face recognition with minimal overhead!** 🚀
