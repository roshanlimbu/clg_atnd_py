"""
========================================================
main.py — Attendance Management System Entry Point
========================================================
STEPS 5–15 (Full Program Loop)

This is the main program. Run this after:
    1. ✅ Teachable Machine model trained and exported to models/
    2. ✅ python convert_model.py  (converts model.json → model.h5)
    3. ✅ python setup.py          (creates DB, installs packages)

Usage:
    python main.py

Controls:
    Q / ESC → Quit the program

What happens:
    - Loads converted Keras model
    - Connects to camera
    - Continuously reads live frames
    - Every 5th frame: detects faces → recognizes → marks attendance
    - Displays live feed with color-coded overlays at all times
    - Runs continuously for days (handles midnight rollover automatically)
========================================================
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Base directory (where main.py lives)
BASE_DIR = Path(__file__).parent

# File paths
MODELS_DIR      = BASE_DIR / "models"
CONVERTED_MODEL = BASE_DIR / "converted_model" / "model.h5"
METADATA_FILE   = MODELS_DIR / "metadata.json"
DATABASE_FILE   = BASE_DIR / "attendance.db"
MEMORY_DIR      = BASE_DIR / "attendance_memory"
LOGS_DIR        = BASE_DIR / "logs"

# Camera settings
CAMERA_INDEX    = 0       # 0 = default device camera
SAMPLE_EVERY    = 5       # Process every 5th frame
DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 720

# Recognition settings
CONFIDENCE_THRESHOLD = 0.85  # 85% — adjust if getting too many false/missed

# Display settings
SHOW_FPS        = True
SHOW_STATUS_BAR = True


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging():
    """Configure logging to both console and daily rotating file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_filename = LOGS_DIR / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_filename), encoding="utf-8"),
        ],
    )

    # Suppress overly verbose library logs
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF C++ logs

    return logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def pre_flight_checks(logger):
    """
    STEP 5 — Verify all required files exist before starting the main loop.
    Exits with a helpful message if anything is missing.
    """
    logger.info("Running pre-flight checks...")
    passed = True

    checks = [
        (CONVERTED_MODEL, "Converted model — run python convert_model.py"),
        (METADATA_FILE,   "Teachable Machine metadata.json in models/"),
        (DATABASE_FILE,   "SQLite database — run python setup.py"),
    ]

    for path, description in checks:
        if path.exists():
            logger.info(f"  ✅ {path.name}")
        else:
            logger.error(f"  ❌ MISSING: {path}")
            logger.error(f"     → {description}")
            passed = False

    if not passed:
        logger.error("\nPre-flight checks failed. Fix the above issues and retry.")
        sys.exit(1)

    logger.info("All pre-flight checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# FPS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class FPSTracker:
    """Simple rolling-average FPS calculator."""

    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self):
        self._times.append(time.time())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("  Face Recognition Attendance Management System")
    logger.info("=" * 60)
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Base dir: {BASE_DIR}")

    # ── Pre-flight checks ────────────────────────────────────────────────
    pre_flight_checks(logger)

    # ── STEP 5 — Initialize all components ──────────────────────────────
    logger.info("Initializing system components...")

    # Database
    from database import DatabaseManager
    db = DatabaseManager(DATABASE_FILE)
    logger.info("✅ Database connected")

    # Memory manager (loads today's file / creates fresh one)
    from memory import MemoryManager
    memory = MemoryManager(MEMORY_DIR)
    logger.info(
        f"✅ Memory loaded — "
        f"{memory.get_marked_count()} persons already marked today"
    )

    # Face detector (YOLOv8n + MediaPipe)
    from detector import FaceDetector
    detector = FaceDetector()
    logger.info("✅ Face detector ready (YOLOv8n + MediaPipe)")

    # Face recognizer (Keras model)
    from recognizer import FaceRecognizer
    recognizer = FaceRecognizer(CONVERTED_MODEL, METADATA_FILE)
    recognizer.set_confidence_threshold(CONFIDENCE_THRESHOLD)
    logger.info(f"✅ Face recognizer ready (threshold={CONFIDENCE_THRESHOLD:.0%})")

    # Attendance recorder
    from attendance import AttendanceRecorder
    recorder = AttendanceRecorder(db, memory)
    logger.info("✅ Attendance recorder ready")

    # Camera feed
    from camera import CameraFeed
    camera = CameraFeed(
        camera_index=CAMERA_INDEX,
        sample_every=SAMPLE_EVERY,
        display_width=DISPLAY_WIDTH,
        display_height=DISPLAY_HEIGHT,
    )

    fps_tracker = FPSTracker()

    logger.info("\nSystem ready. Starting live feed...")
    logger.info("Press Q in the camera window to quit.\n")

    # ── STEP 6 — Start camera feed ───────────────────────────────────────
    if not camera.start():
        logger.error("Failed to open camera. Exiting.")
        sys.exit(1)

    try:
        # Hold the last recognition results so they persist across non-processing frames
        last_detected_faces = []
        last_attendance_results = []

        # ── MAIN LOOP ────────────────────────────────────────────────────
        for frame, should_process in camera.get_frames():
            fps_tracker.tick()

            # ── STEP 7 — Frame sampling ──────────────────────────────────
            if should_process:
                # ── STEP 8 — Multi-face detection (YOLOv8n) ─────────────
                # ── STEP 9 — Face alignment (MediaPipe) ──────────────────
                detected_faces = detector.detect_faces(frame)

                if detected_faces:
                    # ── STEP 10 — Face recognition (Keras model) ─────────
                    # Use batch inference for efficiency with multiple faces
                    face_images = [f.aligned_image for f in detected_faces]
                    recognitions = recognizer.recognize_batch(face_images)

                    # ── STEPS 11+12+13 — Attendance recording ────────────
                    attendance_results = recorder.process_frame_recognitions(
                        recognitions
                    )

                    last_detected_faces = detected_faces
                    last_attendance_results = attendance_results

                    # Log any new markings
                    for result in attendance_results:
                        if result.status == "marked":
                            logger.info(
                                f"🎯 NEW: {result.person_id} ({result.name}) "
                                f"— {result.confidence:.1%} confidence"
                            )
                else:
                    last_detected_faces = []
                    last_attendance_results = []

            # ── STEP 14 — Visual feedback on live feed ───────────────────
            # Convert results to dict format for draw_detections()
            result_dicts = [r.as_dict() for r in last_attendance_results]

            # Draw bounding boxes and labels
            annotated_frame = detector.draw_detections(
                frame, last_detected_faces, result_dicts
            )

            # Draw status bar with stats
            if SHOW_STATUS_BAR:
                annotated_frame = camera.draw_status_bar(
                    annotated_frame,
                    marked_count=memory.get_marked_count(),
                    fps=fps_tracker.fps,
                    processing_frame=should_process,
                )

            # Display the annotated frame
            camera.display_frame(annotated_frame)

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Shutting down...")

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        logger.error(traceback.format_exc())

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        logger.info("\nCleaning up...")
        camera.stop()
        detector.release()

        # Print final summary
        recorder.print_session_summary()

        logger.info("\n" + "=" * 60)
        logger.info("  Attendance system stopped cleanly.")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()