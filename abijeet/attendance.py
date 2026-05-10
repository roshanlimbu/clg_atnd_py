"""
========================================================
attendance.py — Attendance Recording Orchestrator
========================================================
STEP 11 + STEP 12 + STEP 13 + STEP 15

Coordinates the full attendance pipeline for each recognized face:

1. Layer 1 check — memory file (fast, survives restarts)
2. Layer 2 insert — SQLite database (with UNIQUE constraint)
3. Memory update  — write to today's file immediately after DB insert
4. Visual status  — determine color/label for live feed overlay
5. Image Quality  — only save clear, sharp images for evidence
========================================================
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from database import DatabaseManager
from face_identity import FaceIdentityResult
from memory import MemoryManager

logger = logging.getLogger(__name__)


# Status constants for visual overlay
STATUS_MARKED          = "marked"          # Successfully marked (green)
STATUS_ALREADY_MARKED  = "already_marked"  # Debounced inside the time window (yellow)
STATUS_UNKNOWN         = "unknown"         # Unknown face (red)
STATUS_LOW_CONFIDENCE  = "low_confidence"  # Below threshold (gray)
STATUS_INTERNAL        = "internal"        # Internal team member (blue, not counted)
STATUS_NEW_FACE        = "new_face"        # Newly registered person (orange)
STATUS_SILENT_PASS     = "silent_pass"     # Internal team — completely invisible
STATUS_AUTO_REGISTERED = "auto_registered" # Just auto-registered as new visitor
STATUS_UNCERTAIN       = "uncertain"       # In uncertainty buffer, waiting

DEBOUNCE_MINUTES = 5


class ImageQualityChecker:
    """
    Validates image quality before saving attendance evidence.
    Ensures only clear, sharp, well-exposed images are stored.
    """
    
    # Quality thresholds (tuned for face crops)
    MIN_LAPLACIAN_VARIANCE = 25.0  # Lower = blurrier; reject motion blur
    MIN_BRIGHTNESS = 40            # Avoid too-dark images (0-255 scale)
    MAX_BRIGHTNESS = 215           # Avoid too-bright/overexposed images
    MIN_CONTRAST = 30              # Avoid flat/washed out images
    
    @staticmethod
    def is_image_blurry(image: np.ndarray, threshold: float = MIN_LAPLACIAN_VARIANCE) -> bool:
        """
        Detect blur using Laplacian variance (Tenengrad method).
        
        High variance = sharp image
        Low variance = blurry/motion-blurred image
        
        Args:
            image: BGR or RGB image
            threshold: Laplacian variance threshold
            
        Returns:
            True if image is blurry (reject), False if sharp (accept)
        """
        if image is None or image.size == 0:
            return True
        
        try:
            # Convert to grayscale for variance calculation
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Compute Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            is_blurry = laplacian_var < threshold
            logger.debug(f"Blur check: variance={laplacian_var:.1f} (threshold={threshold}) -> {'BLURRY' if is_blurry else 'SHARP'}")
            
            return is_blurry
        except Exception as e:
            logger.debug(f"Blur detection error: {e}")
            return False
    
    @staticmethod
    def get_image_brightness(image: np.ndarray) -> float:
        """
        Calculate average brightness of image (0-255 scale).
        
        Used to reject too-dark or overexposed images.
        """
        if image is None or image.size == 0:
            return 0
        
        try:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            brightness = np.mean(gray)
            return brightness
        except Exception:
            return 127  # Default to neutral
    
    @staticmethod
    def get_image_contrast(image: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation of pixel values.
        
        Used to reject flat/washed-out images.
        """
        if image is None or image.size == 0:
            return 0
        
        try:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            contrast = np.std(gray)
            return contrast
        except Exception:
            return 0
    
    @staticmethod
    def check_image_quality(
        image: np.ndarray,
        person_id: str = "unknown",
        confidence: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Comprehensive image quality check.
        
        Args:
            image: Face crop image to validate
            person_id: Person ID (for logging)
            confidence: Recognition confidence (for logging)
            
        Returns:
            (is_valid: bool, reason: str)
            - is_valid=True if image passes all checks
            - reason explains why image was rejected (if applicable)
        """
        if image is None or image.size == 0:
            return False, "Image is empty"
        
        if image.ndim != 3 or image.shape[2] != 3:
            return False, f"Invalid image shape: {image.shape}"
        
        # Check 1: Blur detection
        if ImageQualityChecker.is_image_blurry(image):
            return False, f"Image too blurry (motion detected) - REJECTED for {person_id}"
        
        # Check 2: Brightness
        brightness = ImageQualityChecker.get_image_brightness(image)
        if brightness < ImageQualityChecker.MIN_BRIGHTNESS:
            return False, f"Image too dark (brightness={brightness:.0f}) - REJECTED for {person_id}"
        if brightness > ImageQualityChecker.MAX_BRIGHTNESS:
            return False, f"Image overexposed (brightness={brightness:.0f}) - REJECTED for {person_id}"
        
        # Check 3: Contrast
        contrast = ImageQualityChecker.get_image_contrast(image)
        if contrast < ImageQualityChecker.MIN_CONTRAST:
            return False, f"Image too flat/low contrast (contrast={contrast:.0f}) - REJECTED for {person_id}"
        
        # All checks passed
        logger.info(
            f"✅ Image quality OK for {person_id} | "
            f"Sharpness={cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var():.1f}, "
            f"Brightness={brightness:.0f}, Contrast={contrast:.0f}"
        )
        return True, "Image quality passed all checks"


class AttendanceResult:
    """Result of processing one detected face through the attendance pipeline."""

    def __init__(
        self,
        person_id: str,
        confidence: float,
        status: str,
        count: int = 0,
        display_name: Optional[str] = None,
    ):
        self.person_id = person_id
        self.confidence = confidence
        self.status = status
        self.count = count
        self.display_name = display_name

    def as_dict(self) -> dict:
        """Convert to dict for use with detector.draw_detections()."""
        return {
            "person_id": self.person_id,
            "confidence": self.confidence,
            "status": self.status,
            "count": self.count,
            "display_name": self.display_name,
        }

    def __repr__(self):
        return (
            f"AttendanceResult("
            f"person_id={self.person_id!r}, "
            f"confidence={self.confidence:.2%}, "
            f"status={self.status!r}, "
            f"count={self.count!r})"
        )


class AttendanceRecorder:
    """
    STEP 11 + 12 + 13 + 15 — Orchestrates the full attendance pipeline.

    Takes recognition results and coordinates:
        - Duplicate checking (Layer 1: memory file)
        - Database recording (Layer 2: UNIQUE constraint backup)
        - Memory file updating
        - Midnight rollover for multi-day continuous operation
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        memory_manager: MemoryManager,
        photos_dir: Optional[Path] = None,
    ):
        """
        Args:
            db_manager    : Initialized DatabaseManager instance
            memory_manager: Initialized MemoryManager instance
        """
        self.db = db_manager
        self.memory = memory_manager
        self.photos_dir = photos_dir
        if self.photos_dir is not None:
            self.photos_dir.mkdir(parents=True, exist_ok=True)

        # Session statistics
        self._session_marked_count: int = 0
        self._session_duplicate_count: int = 0
        self._session_unknown_count: int = 0

        logger.info("AttendanceRecorder initialized.")

    def process_recognition(
        self,
        recognition: FaceIdentityResult,
        face_image: Optional[np.ndarray] = None,
    ) -> AttendanceResult:
        """
        STEPS 11 + 12 + 13 — Process a single face recognition result.

        Pipeline:
        1. Unknown/low confidence → return immediately with appropriate status
        2. Layer 1 check (memory file) → already marked? return already_marked
        3. Layer 2 insert (database) → insert with UNIQUE constraint
        4. Memory update → write to today's file immediately
        5. Return status for visual overlay

        Args:
            recognition: FaceIdentityResult from FaceIdentityManager

        Returns:
            AttendanceResult with final status for display
        """
        # ── Handle Unknown / Low Confidence ──────────────────────────────
        if recognition.is_unknown:
            self._session_unknown_count += 1
            return AttendanceResult(
                person_id="Unknown",
                confidence=recognition.confidence,
                status=STATUS_UNKNOWN,
            )

        if not recognition.is_above_threshold:
            return AttendanceResult(
                person_id=recognition.person_id,
                confidence=recognition.confidence,
                status=STATUS_LOW_CONFIDENCE,
                display_name=recognition.display_name,
            )

        person_id = recognition.person_id
        confidence = recognition.confidence

        if recognition.is_internal:
            logger.debug(
                "Internal team member ignored for attendance: %s",
                recognition.display_name or person_id,
            )
            return AttendanceResult(
                person_id=person_id,
                confidence=confidence,
                status=STATUS_INTERNAL,
                display_name=recognition.display_name,
            )

        # ── STEP 12 — Record in SQLite Database ──────────────────────────
        now = datetime.now()
        db_action, count = self.db.record_attendance(
            person_id=person_id,
            confidence=confidence,
            attendance_date=now.date(),
            attendance_time=now,
            debounce_minutes=DEBOUNCE_MINUTES,
        )

        if db_action == "debounced":
            self._session_duplicate_count += 1
            return AttendanceResult(
                person_id=person_id,
                confidence=confidence,
                status=STATUS_ALREADY_MARKED,
                count=count,
                display_name=recognition.display_name,
            )

        if db_action == "error":
            return AttendanceResult(
                person_id=person_id,
                confidence=confidence,
                status=STATUS_LOW_CONFIDENCE,
                count=count,
                display_name=recognition.display_name,
            )

        # ── STEP 13 — Update Memory File ─────────────────────────────────
        photo_path = self._save_attendance_photo(
            person_id=person_id,
            captured_at=now,
            count=count,
            confidence=confidence,
            face_image=face_image,
        )
        if photo_path is not None:
            self.db.record_attendance_photo(
                person_id=person_id,
                attendance_date=now.date(),
                attendance_time=now,
                count=count,
                confidence=confidence,
                image_path=photo_path,
            )
            # For newly registered persons, store the first photo as reference
            if recognition.is_new:
                self.db.update_person_reference_photo(person_id, photo_path)

        self.memory.mark_person(person_id)
        self._session_marked_count += 1

        status = STATUS_NEW_FACE if recognition.is_new else STATUS_MARKED
        logger.info(
            "COUNTED: %s | %s | count=%s | confidence=%.2f%% | time=%s | photo=%s",
            person_id,
            "NEW REGISTRATION" if recognition.is_new else "returning",
            count, confidence * 100,
            now.strftime("%H:%M:%S"), photo_path or "not saved",
        )

        return AttendanceResult(
            person_id=person_id,
            confidence=confidence,
            status=status,
            count=count,
            display_name=recognition.display_name,
        )

    def process_frame_recognitions(
        self,
        recognitions: List[FaceIdentityResult],
        face_images: Optional[List[np.ndarray]] = None,
    ) -> List[AttendanceResult]:
        """
        Process all recognition results from a single frame.

        Args:
            recognitions: List of FaceIdentityResult objects (one per detected face)

        Returns:
            List of AttendanceResult objects (same order as input)
        """
        # STEP 15 — Check for midnight rollover before processing each frame
        date_changed = self.memory.check_and_refresh_date()
        if date_changed:
            logger.info(
                "New day detected during frame processing. "
                "Attendance memory reset. All persons can be marked again."
            )

        results = []
        for index, recognition in enumerate(recognitions):
            face_image = None
            if face_images is not None and index < len(face_images):
                face_image = face_images[index]
            result = self.process_recognition(recognition, face_image)
            results.append(result)

        return results

    def _save_attendance_photo(
        self,
        person_id: str,
        captured_at: datetime,
        count: int,
        confidence: float,
        face_image: Optional[np.ndarray],
    ) -> Optional[str]:
        """
        Save the face crop for one counted attendance event.
        
        Quality checks ensure only clear, sharp images are stored as evidence.
        Blurry or poorly exposed images are rejected and not saved.
        """
        if self.photos_dir is None or face_image is None or face_image.size == 0:
            return None

        # Quality check BEFORE saving
        is_valid, reason = ImageQualityChecker.check_image_quality(
            face_image, 
            person_id=person_id,
            confidence=confidence
        )
        
        if not is_valid:
            logger.warning(
                f"⚠️  Image rejected for {person_id} (conf={confidence:.2%}): {reason}"
            )
            return None

        try:
            date_dir = self.photos_dir / captured_at.date().isoformat()
            date_dir.mkdir(parents=True, exist_ok=True)
            timestamp = captured_at.strftime("%H%M%S")
            safe_person_id = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in person_id
            )
            
            # Add quality indicators to filename
            laplacian_var = cv2.Laplacian(
                cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY),
                cv2.CV_64F
            ).var()
            brightness = ImageQualityChecker.get_image_brightness(face_image)
            
            filename = (
                f"{timestamp}_{safe_person_id}_count-{count}_"
                f"conf-{int(confidence * 100):03d}_"
                f"sharp-{int(laplacian_var):03d}.jpg"
            )
            output_path = date_dir / filename

            image = face_image
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            if image.ndim == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Use high quality JPEG compression
            success = cv2.imwrite(
                str(output_path), 
                image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95]  # 95/100 quality to preserve details
            )
            
            if not success:
                logger.warning("OpenCV did not save attendance photo: %s", output_path)
                return None

            logger.info(
                f"📸 HIGH-QUALITY photo saved: {person_id} | "
                f"Sharpness={laplacian_var:.0f} | Brightness={brightness:.0f} | "
                f"{output_path.name}"
            )
            return output_path.relative_to(self.photos_dir.parent).as_posix()
        except Exception as exc:
            logger.warning("Failed to save attendance photo for %s: %s", person_id, exc)
            return None

    def get_session_stats(self) -> dict:
        """Return statistics for this program session."""
        memory_status = self.memory.get_status_summary()
        db_stats = self.db.get_statistics()

        return {
            # Session (since last restart)
            "session_marked": self._session_marked_count,
            "session_duplicates_blocked": self._session_duplicate_count,
            "session_unknowns": self._session_unknown_count,

            # Today (all-time today, from DB)
            "today_total": db_stats["today_count"],
            "today_date": memory_status["today"],

            # All-time
            "all_time_records": db_stats["all_time_records"] if "all_time_records" in db_stats else db_stats.get("all_time_count", 0),
            "registered_persons": db_stats["total_persons"],
        }

    def print_session_summary(self):
        """Print a formatted summary of today's attendance to the log."""
        stats = self.get_session_stats()
        records = self.db.get_today_attendance()

        logger.info("=" * 55)
        logger.info(f"  ATTENDANCE SUMMARY — {stats['today_date']}")
        logger.info("=" * 55)
        logger.info(f"  Attendance count     : {stats['today_total']}")
        logger.info(f"  Unique faces         : {stats['registered_persons']}")
        logger.info(f"  Session counted      : {stats['session_marked']}")
        logger.info(f"  Debounced sightings  : {stats['session_duplicates_blocked']}")
        logger.info(f"  Unknown faces seen   : {stats['session_unknowns']}")
        logger.info("-" * 55)

        if records:
            logger.info("  Attendance log:")
            for rec in records:
                logger.info(
                    f"    {rec['first_seen']}  {rec['person_id']:<20}  "
                    f"count={rec['count']:<3}  last={rec['last_seen']}"
                )
        else:
            logger.info("  No attendance records yet today.")

        logger.info("=" * 55)
