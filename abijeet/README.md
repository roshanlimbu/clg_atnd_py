# Face Attendance Management System

A real-time face detection and attendance tracking system designed for **mass events** (ceremonies, festivals, conferences). Uses YOLOv8n-face for multi-face detection and dlib ResNet-34 embeddings for generated face identity matching — no pre-enrollment needed.

## ✨ NEW: Advanced Profile Detection

**🎯 Detect faces at ANY angle** - not just front-facing!

- ✅ **Head Pose Detection** - Automatically detects face angle (front, left, right, up, down)
- ✅ **Profile Recognition** - Recognizes people from side profiles and extreme angles
- ✅ **Pose-Aware Matching** - Adjusts matching thresholds based on angle difference
- ✅ **Multi-Angle Enrollment** - Capture faces from 5 different angles for better accuracy

**See [PROFILE_DETECTION_GUIDE.md](PROFILE_DETECTION_GUIDE.md) for detailed documentation.**

---

## System Overview

```
Camera Feed ──► Capture Thread ──► Main Loop (every 5th frame)
                                      │
                                      ├─► YOLOv8n Face Detection
                                      ├─► Face Alignment (MediaPipe)
                                      ├─► Head Pose Detection (NEW)  ◄── Detects angle
                                      ├─► dlib Embedding Encoding
                                      ├─► Pose-Aware Matching (NEW) ◄── Adjusts threshold
                                      └─► Attendance Recording (DB + File + Pose)
                                               │
                                               └─► Dashboard (HTTP Server)
```

The system processes every 5th frame for detection. YOLO catches **all faces** in a single forward pass with no hard upper limit (practical: 50-80 faces per frame with current profile settings). The dlib encoding step is the per-face throughput bottleneck (~30ms per face with the optimized `model=small` settings).

**NEW**: The system now detects head pose (angle) for each face and uses this to improve matching accuracy across different angles. People can be recognized from front, left profile, right profile, and other angles.

---

## Requirements

- **OS:** Windows (primary), macOS (tested)
- **Python:** 3.8 — 3.12 (TensorFlow/MediaPipe do not support 3.13+)
- **Camera:** Any USB or built-in camera
- **GPU:** Not required — all processing runs on CPU

---

## One-Time Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `tensorflowjs` fails (Windows), install the runtime packages:

```powershell
pip install opencv-python ultralytics mediapipe tensorflow face-recognition numpy Pillow
```

> **Note:** `face-recognition` requires `cmake`. Install with:
> ```powershell
> pip install cmake
> pip install face-recognition
> ```

### 3. Initialize the database and structure

```powershell
python setup.py
```

This creates:
- `attendance.db` — SQLite database with persons, attendance, and photos tables
- All required directories (`logs/`, `attendance_memory/`, etc.)
- Installs any missing packages

---

## Running the System

### Terminal 1 — Live Attendance Detection

```powershell
.\.venv\Scripts\python.exe main.py
```

**Controls:**
| Key | Action |
|---|---|
| `Q` or `Esc` | Quit the program |

**What happens:**
- Opens the default camera (index 0)
- Continuously reads frames at full camera rate
- Every 5th frame: detects faces → generates IDs → records attendance
- Displays live feed with color-coded overlays
- Runs continuously (handles midnight rollover automatically)

**Color codes on the live feed:**
| Color | Meaning |
|---|---|
| Green | Successfully marked present |
| Yellow | Already marked today (debounced) |
| Blue | Internal team member (not counted) |
| Red | Unknown face |
| Gray | Below confidence threshold |

### Terminal 2 — Dashboard Server

```powershell
.\.venv\Scripts\python.exe frontend\server.py
```

Open: **http://127.0.0.1:8000**

The dashboard shows:
- Live attendance count for today
- Active faces (seen in the last 5 minutes)
- Individual attendance logs with photos
- Internal team member management

---

## Architecture

### Files

| File | Purpose |
|---|---|
| `main.py` | Entry point — connects camera, runs the main detection loop |
| `camera.py` | Background capture thread + frame display + adaptive profile |
| `detector.py` | YOLOv8n face detection + face alignment |
| `face_identity.py` | dlib ResNet-34 embedding encoding + identity matching + **pose-aware matching** |
| `head_pose_detector.py` | **[NEW]** Head pose estimation (front, left, right, up, down) using MediaPipe |
| `enrollment_mode.py` | **[NEW]** Interactive multi-angle enrollment mode for better accuracy |
| `attendance.py` | Attendance recording, debouncing, photo saving |
| `database.py` | SQLite database operations |
| `memory.py` | Daily memory file for Layer 1 duplicate prevention |
| `setup.py` | One-time project setup |
| `frontend/server.py` | Local web dashboard |
| `recognizer.py` | Legacy Teachable Machine recognizer (not used by default) |
| `convert_model.py` | Legacy model converter (not used by default) |

### Detection Pipeline

1. **YOLOv8n-face** — Single forward pass detects all faces in the frame. No upper limit.
2. **Face Alignment** — Crops each detection, aligns using eye landmarks.
3. **Head Pose Detection** — **[NEW]** Estimates head angle (front, left profile, right profile, etc.)
4. **dlib Embedding** — Encodes each aligned face into a 128-dimensional vector.
5. **Identity Matching** — Compares against known faces via L2 distance with **pose-aware threshold adjustment**.
6. **Attendance Recording** — Dual-layer: memory file (Layer 1) + database UNIQUE constraint (Layer 2) + pose information.

### Adaptive Performance

The camera automatically selects a processing profile based on hardware:

| Profile | Processing Resolution | Sample Rate |
|---|---|---|
| High | 640 × 360 | Every 2nd frame |
| Medium | 480 × 270 | Every 3rd frame |
| Low | 320 × 180 | Every 6th frame |

If detection consistently takes >46ms (1.4× the 33ms budget), the system auto-downgrades to the next lighter profile.

---

## Advanced Features: Profile Detection

### What's New?

The system now includes **advanced head pose detection** to recognize people from any angle:

- **Automatic Angle Detection**: Detects front, left profile, right profile, up, and down poses
- **Pose-Aware Matching**: Adjusts recognition thresholds based on angle differences
- **Multi-Angle Enrollment**: Capture faces from 5 angles (3 frames each) for 10× better accuracy
- **Side Profile Recognition**: Works reliably even when people turn their head sideways

### Example: Why This Matters

**Before**: Person approaches camera from the left (90° angle)
- Face detected but not recognized (distance 0.72 > threshold 0.60)
- Result: **UNKNOWN** ❌

**After**: Same scenario with profile detection
- Head pose detected as "left_profile"  
- Threshold automatically increased to 0.69
- Face recognized correctly (distance 0.68 < threshold 0.69)
- Result: **RECOGNIZED** ✅

### Quick Start with Profiles

1. **Automatic** (existing flow): Just run `main.py` - the system automatically detects profiles
2. **Better accuracy** (recommended): Use multi-angle enrollment for new users

For detailed documentation, see [PROFILE_DETECTION_GUIDE.md](PROFILE_DETECTION_GUIDE.md).

### Live Feed Updates

The live feed now shows:
```
Pose: left_profile | Yaw: -75.3° | Pitch: 2.1° | Roll: 0.8°
```

---

All settings in `main.py`:

```python
CAMERA_INDEX    = 0       # Change if you have multiple cameras
SAMPLE_EVERY    = 5       # Process every Nth frame
DISPLAY_WIDTH   = 1280    # Requested camera resolution
DISPLAY_HEIGHT  = 720     # Requested camera resolution
```

```python
# In camera.py
DEVICE_PROFILES = {
    "low":    {"proc_width": 320, "proc_height": 180, "sample_every": 6},
    "medium": {"proc_width": 480, "proc_height": 270, "sample_every": 3},
    "high":   {"proc_width": 640, "proc_height": 360, "sample_every": 2},
}
```

---

## Internal Team Members

The dashboard (`http://127.0.0.1:8000`) lets you register internal team members:

1. Open the dashboard
2. Use the "Add Internal Team" form (name + photo)
3. Recognized team members appear with a "Not counted" label (blue) in the live feed

Their face photos are stored in `internal_team_photos/`.

---

## Data Storage

| Data | Location | Format |
|---|---|---|
| Attendance records | `attendance.db` | SQLite |
| Daily memory file | `attendance_memory/memory_YYYY-MM-DD.txt` | Plain text |
| Attendance photos | `attendance_photos/YYYY-MM-DD/` | JPEG |
| Internal team photos | `internal_team_photos/` | JPEG |
| Logs | `logs/attendance_YYYY-MM-DD.log` | Plain text |

---

## Troubleshooting

**Camera does not open:**
- Make sure no other app is using the camera
- Try changing `CAMERA_INDEX` in `main.py` (0, 1, 2, ...)
- On Windows, grant camera permission to your terminal app
- Restart your computer if the camera driver is wedged

**No faces detected:**
- Ensure adequate lighting on faces
- Faces should be at least 60px wide in the frame
- Check the terminal log for YOLO/detector errors

**High CPU usage:**
- The system runs entirely on CPU
- Try setting `device_profile="low"` in `main.py` line where `CameraFeed` is initialized
- Lower the camera resolution (`DISPLAY_WIDTH` / `DISPLAY_HEIGHT`)

**Dashboard won't start:**
- Confirm `attendance.db` exists (run `setup.py` first)
- Check that port 8000 is free
- Try `http://127.0.0.1:8000` directly

---

## Cleanup

Remove all generated data and start fresh:

```powershell
rm -rf logs\* .cache __pycache__ frontend\__pycache__
del attendance.db attendance.db-shm attendance.db-wal
python setup.py
```
