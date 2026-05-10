"""
========================================================
main.py — Attendance Management System Entry Point
========================================================
STEPS 5–15 (Full Program Loop)

This is the main program. Run this after:
    1. ✅ python setup.py          (creates DB, installs packages)

Usage:
    python main.py

Controls:
    Q / ESC → Quit the program

What happens:
    - Connects to camera
    - Continuously reads live frames
    - Every 5th frame: detects faces → assigns generated IDs → counts attendance
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

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Base directory (where main.py lives)
BASE_DIR = Path(__file__).parent

# File paths
MODELS_DIR = BASE_DIR / "models"
CONVERTED_MODEL = BASE_DIR / "converted_model" / "model.h5"
METADATA_FILE = MODELS_DIR / "metadata.json"
DATABASE_FILE = BASE_DIR / "attendance.db"
MEMORY_DIR = BASE_DIR / "attendance_memory"
LOGS_DIR = BASE_DIR / "logs"
PHOTOS_DIR = BASE_DIR / "attendance_photos"

# Camera settings
CAMERA_INDEX = 0  # 0 = default device camera
SAMPLE_EVERY = 5  # Process every 5th frame
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Display settings
SHOW_FPS = True
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

    major, minor = sys.version_info[:2]
    if major > 3 or (major == 3 and minor >= 13):
        logger.error(
            "Python %s.%s is too new for TensorFlow/MediaPipe. "
            "Use Python 3.12 for this project.",
            major,
            minor,
        )
        passed = False

    checks = [
        (DATABASE_FILE, "SQLite database — run python setup.py"),
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


def _scale_faces_for_display(detected_faces, src_shape, dst_shape):
    """Scale detection boxes from the source frame size to the display size."""
    if not detected_faces:
        return []

    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if src_w == dst_w and src_h == dst_h:
        return detected_faces

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    scaled = []
    for face in detected_faces:
        x1, y1, x2, y2 = face.bbox
        nx1 = max(0, min(dst_w, int(round(x1 * scale_x))))
        ny1 = max(0, min(dst_h, int(round(y1 * scale_y))))
        nx2 = max(0, min(dst_w, int(round(x2 * scale_x))))
        ny2 = max(0, min(dst_h, int(round(y2 * scale_y))))
        scaled.append(
            type(face)(
                bbox=(nx1, ny1, nx2, ny2),
                aligned_image=face.aligned_image,
                detection_confidence=face.detection_confidence,
            )
        )
    return scaled


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
# RESULT.MD GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def write_result_md(db, recorder, output_path=None):
    """Write a summarized attendance report to result.md."""
    if output_path is None:
        output_path = BASE_DIR / "result.md"

    stats = recorder.get_session_stats()
    records = db.get_today_attendance()
    all_persons = db.get_all_persons()
    now = datetime.now()

    lines = []
    lines.append("# Attendance System — Session Report")
    lines.append("")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Date:** {stats['today_date']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total attendance counts today | {stats['today_total']} |")
    lines.append(f"| Registered persons (all-time) | {stats['registered_persons']} |")
    lines.append(f"| Session: faces counted | {stats['session_marked']} |")
    lines.append(
        f"| Session: debounced sightings | {stats['session_duplicates_blocked']} |"
    )
    lines.append(f"| Session: unknown faces | {stats['session_unknowns']} |")
    lines.append("")

    if records:
        lines.append("## Today's Attendance Log")
        lines.append("")
        lines.append("| Person ID | Count | First Seen | Last Seen | Display Name |")
        lines.append("|-----------|-------|------------|-----------|--------------|")
        for rec in records:
            display = rec["display_name"] or "-"
            lines.append(
                f"| {rec['person_id']} | {rec['count']} | "
                f"{rec['first_seen']} | {rec['last_seen']} | {display} |"
            )
        lines.append("")

    if all_persons:
        lines.append("## All Registered Persons")
        lines.append("")
        lines.append("| Person ID | Role | Registered Date | Display Name |")
        lines.append("|-----------|------|-----------------|--------------|")
        for p in all_persons:
            role = p["role"] or "guest"
            reg_date = str(p["registered_date"]) if p["registered_date"] else "-"
            display = p["display_name"] or "-"
            lines.append(f"| {p['person_id']} | {role} | {reg_date} | {display} |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


logger = logging.getLogger(__name__)

PENDING_FACES_DIR = PHOTOS_DIR / "pending"


def _save_pending_face(track, db, photos_dir, face_tracker):
    """
    Save a confirmed new face crop to disk for frontend review.

    Instead of auto-registering, we:
    1. Save the best quality crop to attendance_photos/pending/
    2. Record metadata in the pending_faces table
    3. Mark the track as registered (to prevent re-capture)
    The frontend operator then decides to register or dismiss.
    """
    import cv2

    if track.best_crop is None:
        return

    pending_dir = photos_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pending_{timestamp}_track{track.track_id}.jpg"
    filepath = pending_dir / filename
    relative_path = filepath.relative_to(photos_dir.parent).as_posix()

    # Save the best crop
    try:
        crop_bgr = cv2.cvtColor(track.best_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), crop_bgr)
    except Exception as e:
        logger.error("Failed to save pending face crop: %s", e)
        return

    # Record in DB for frontend to pick up
    db.record_pending_face(
        image_path=relative_path,
        quality_score=track.best_quality,
        best_similarity=track.best_raw_similarity,
        frames_seen=track.frames_seen,
    )

    # Mark track so we don't recapture it
    from face_identity import FaceIdentityResult

    dummy_result = FaceIdentityResult(
        person_id="pending",
        confidence=0.0,
        zone="registered",
    )
    face_tracker.mark_registered(track.track_id, dummy_result)

    logger.info(
        "PENDING FACE saved: %s (seen %d frames, quality=%.1f, best_sim=%.3f)",
        filename,
        track.frames_seen,
        track.best_quality,
        track.best_raw_similarity,
    )


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
    db.ensure_auto_registration_table()  # Create auto_registered_visitors if needed
    db.ensure_pending_faces_table()  # Create pending_faces for frontend registration
    logger.info("Database connected")

    # Memory manager (loads today's file / creates fresh one)
    from memory import MemoryManager

    memory = MemoryManager(MEMORY_DIR)
    logger.info("✅ Daily memory loaded")

    # Face detector (YOLOv8n + MediaPipe)
    from detector import FaceDetector

    detector = FaceDetector()
    logger.info("✅ Face detector ready")

    # Face identity manager (generated IDs; no fixed trained labels)
    from face_identity import FaceIdentityManager

    identity_manager = FaceIdentityManager(db)
    logger.info("✅ Face identity manager ready")

    # Attendance recorder
    from attendance import AttendanceRecorder

    recorder = AttendanceRecorder(db, memory, PHOTOS_DIR)
    logger.info("✅ Attendance recorder ready")

    # Camera feed — use the newer CameraFeed API (capture resolution,
    # device profile). The feed itself draws the status bar onto frames,
    # so we don't need to draw it here.
    from camera import CameraFeed

    camera = CameraFeed(
        camera_index=CAMERA_INDEX,
        capture_width=DISPLAY_WIDTH,
        capture_height=DISPLAY_HEIGHT,
        device_profile="auto",
        fullscreen=True,
    )

    fps_tracker = FPSTracker()

    # Face tracker — tracks faces across frames, recognizes once per person
    from tracker import FaceTracker

    face_tracker = FaceTracker()

    logger.info("\nSystem ready. Starting live feed...")
    logger.info("Press Q in the camera window to quit.\n")

    # ── Sharp frame buffer for quality image selection ──────────────────
    from frame_buffer import SharpFrameBuffer

    sharp_buffer = SharpFrameBuffer()

    # ── STEP 6 — Start camera feed ───────────────────────────────────────
    if not camera.start():
        logger.error("Failed to open camera. Exiting.")
        sys.exit(1)

    try:
        # Hold the last recognition results so they persist across non-processing frames
        last_detected_faces = []
        last_attendance_results = []

        # ── MAIN LOOP ────────────────────────────────────────────────────
        for frame, process_frame in camera.get_frames():
            fps_tracker.tick()

            # ── STEP 7 — Frame sampling ──────────────────────────────────
            # `process_frame` is either a downscaled ndarray (to run inference)
            # or `None` for non-sampled frames. Use explicit None check to avoid
            # treating numpy arrays as booleans (which raises an error).
            if process_frame is not None:
                # ── STEP 8 — Multi-face detection (YOLOv8n) ─────────────
                # ── STEP 9 — Face alignment (MediaPipe) ──────────────────
                # Prefer using the smaller `process_frame` for detection when
                # available to reduce CPU load. Fall back to the full display
                # frame if needed. Measure total processing time and report
                # it to the camera so adaptive downgrades can occur.
                detection_input = process_frame
                t0 = time.perf_counter()
                detected_faces = detector.detect_faces(detection_input)
                display_faces = _scale_faces_for_display(
                    detected_faces,
                    detection_input.shape,
                    frame.shape,
                )

                if detected_faces:
                    # ── STEP 10a — Update tracker with new detections ────
                    face_tracker.update(detected_faces)

                    # ── STEP 10b — Recognize only qualified tracks ───────
                    tracks_to_recognize = face_tracker.get_tracks_to_recognize()
                    if tracks_to_recognize:
                        face_images = [t.best_crop for t in tracks_to_recognize]
                        new_recognitions = identity_manager.identify_batch(face_images)
                        for track, recog in zip(tracks_to_recognize, new_recognitions):
                            face_tracker.store_recognition(track.track_id, recog)

                    # ── STEP 10c — Auto-register genuinely new faces ─────────
                    # This is the CORE of the unknown-tracking system:
                    # - Internal team = known → silent pass (excluded)
                    # - Random unknown people → confirmed over 3+ frames →
                    #   auto-registered as a new guest AND attendance recorded NOW.
                    tracks_to_register = face_tracker.get_tracks_to_register()
                    for track in tracks_to_register:
                        result = identity_manager.register_new_face(track.best_crop)
                        if result is not None:
                            face_tracker.mark_registered(track.track_id, result)
                            logger.info(
                                "NEW UNKNOWN PERSON registered and counted: %s (Track %d)",
                                result.person_id,
                                track.track_id,
                            )
                            # ── Record first attendance for this new unknown person ──
                            # We do this immediately here because on the next frames the
                            # track will be zone='registered' with above_threshold=True
                            # and will continue to be debounce-counted normally.
                            first_hit = recorder.process_recognition(
                                result,
                                track.best_crop,
                            )
                            logger.info(
                                "First attendance recorded for %s: status=%s, count=%s",
                                result.person_id,
                                first_hit.status,
                                first_hit.count,
                            )

                    logger.debug(
                        "Tracker: %d active, %d silent, %d uncertain, %d pending_reg, %d recognized",
                        face_tracker.active_count,
                        face_tracker.silent_pass_count,
                        face_tracker.uncertain_count,
                        face_tracker.pending_registration_count,
                        len(tracks_to_recognize),
                    )

                    # ── STEPS 11+12+13 — Attendance (skip internal team) ──
                    # Show ALL active tracks on screen (including internal)
                    # but only record attendance for non-internal people
                    from attendance import AttendanceResult, STATUS_INTERNAL

                    all_active = face_tracker.get_active_tracks()

                    # Separate internal from non-internal for attendance
                    recognitions = []
                    face_images_for_attendance = []
                    internal_tracks = []

                    for track in all_active:
                        if track.zone == "silent_pass":
                            # Internal team — detected but NEVER counted
                            internal_tracks.append(track)
                            continue
                        if track.recognition_result is not None:
                            # Skip uncertain/discarded — not ready yet
                            if track.zone in ("uncertain", "discarded", "pending"):
                                continue
                            # Skip new_face zone — not registered yet,
                            # will be handled by tracks_to_register above.
                            if track.zone == "new_face":
                                continue
                            # 'matched' or 'registered' zone → count attendance
                            # For registered: above_threshold=True, unknown=False, is_new=True
                            # For matched: normal returning unknown guest
                            recognitions.append(track.recognition_result)
                            face_images_for_attendance.append(
                                track.best_crop
                                if track.best_crop is not None
                                else track.current_crop
                            )

                    if recognitions:
                        attendance_results = recorder.process_frame_recognitions(
                            recognitions,
                            face_images_for_attendance,
                        )
                    else:
                        attendance_results = []

                    # ── Build display for ALL tracks (including internal) ──
                    from detector import DetectedFace

                    display_faces = []
                    combined_results = []

                    # First: non-internal tracks (have real attendance results)
                    result_idx = 0
                    for track in all_active:
                        if track.zone == "silent_pass":
                            continue  # Handle below
                        display_faces.append(
                            DetectedFace(
                                bbox=track.bbox,
                                aligned_image=(
                                    track.current_crop
                                    if track.current_crop is not None
                                    else np.zeros((224, 224, 3), dtype=np.uint8)
                                ),
                                detection_confidence=track.detection_confidence,
                            )
                        )
                        # Match attendance result to this track
                        if (
                            track.recognition_result is not None
                            and track.zone not in ("uncertain", "discarded", "pending")
                            and result_idx < len(attendance_results)
                        ):
                            combined_results.append(attendance_results[result_idx])
                            result_idx += 1
                        else:
                            # Track without attendance result (uncertain/pending)
                            combined_results.append(
                                AttendanceResult(
                                    person_id=(
                                        getattr(
                                            track.recognition_result, "person_id", "..."
                                        )
                                        if track.recognition_result
                                        else "..."
                                    ),
                                    confidence=track.detection_confidence,
                                    status="low_confidence",
                                )
                            )

                    # Then: internal team tracks (blue box, "Not counted")
                    for track in internal_tracks:
                        display_faces.append(
                            DetectedFace(
                                bbox=track.bbox,
                                aligned_image=(
                                    track.current_crop
                                    if track.current_crop is not None
                                    else np.zeros((224, 224, 3), dtype=np.uint8)
                                ),
                                detection_confidence=track.detection_confidence,
                            )
                        )
                        recog = track.recognition_result
                        combined_results.append(
                            AttendanceResult(
                                person_id=(
                                    getattr(recog, "person_id", "Internal")
                                    if recog
                                    else "Internal"
                                ),
                                confidence=(
                                    getattr(recog, "confidence", 0.0) if recog else 0.0
                                ),
                                status=STATUS_INTERNAL,
                                display_name=(
                                    getattr(recog, "display_name", None)
                                    if recog
                                    else None
                                ),
                            )
                        )

                    # Scale to display coordinates
                    display_faces = _scale_faces_for_display(
                        display_faces,
                        detection_input.shape,
                        frame.shape,
                    )

                    last_detected_faces = display_faces
                    last_attendance_results = combined_results

                    # Log any new markings
                    for result in combined_results:
                        if result.status == "marked":
                            logger.info(
                                "COUNTED: %s — count=%s",
                                result.person_id,
                                result.count,
                            )
                        elif result.status == "new_face":
                            logger.info(
                                "NEW VISITOR COUNTED: %s — auto-registered",
                                result.person_id,
                            )
                else:
                    face_tracker.update([])  # Track disappearances
                    last_detected_faces = []
                    last_attendance_results = []

                # Report processing time (ms) to the camera for adaptive logic
                t1 = time.perf_counter()
                camera.record_process_time((t1 - t0) * 1000)

            # ── STEP 14 — Visual feedback on live feed ───────────────────
            # Convert results to dict format for draw_detections()
            result_dicts = [r.as_dict() for r in last_attendance_results]

            # Draw bounding boxes and labels
            annotated_frame = detector.draw_detections(
                frame, last_detected_faces, result_dicts
            )

            # The camera feed already draws a status bar onto `display` frames
            # before yielding them, so we only need to display the annotated
            # result here.
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
        sharp_buffer.clear_all()

        # Print final summary
        recorder.print_session_summary()

        # Write result.md report
        result_path = write_result_md(db, recorder)
        logger.info("Attendance report written to: %s", result_path)

        logger.info("\n" + "=" * 60)
        logger.info("  Attendance system stopped cleanly.")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
