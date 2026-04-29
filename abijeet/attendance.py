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
========================================================
"""

import logging
from datetime import datetime
from typing import List

from database import DatabaseManager
from face_identity import FaceIdentityResult
from memory import MemoryManager

logger = logging.getLogger(__name__)


# Status constants for visual overlay
STATUS_MARKED          = "marked"          # Successfully marked (green)
STATUS_ALREADY_MARKED  = "already_marked"  # Debounced inside the time window (yellow)
STATUS_UNKNOWN         = "unknown"         # Unknown face (red)
STATUS_LOW_CONFIDENCE  = "low_confidence"  # Below threshold (gray)

DEBOUNCE_MINUTES = 5


class AttendanceResult:
    """Result of processing one detected face through the attendance pipeline."""

    def __init__(
        self,
        person_id: str,
        confidence: float,
        status: str,
        count: int = 0,
    ):
        self.person_id = person_id
        self.confidence = confidence
        self.status = status
        self.count = count

    def as_dict(self) -> dict:
        """Convert to dict for use with detector.draw_detections()."""
        return {
            "person_id": self.person_id,
            "confidence": self.confidence,
            "status": self.status,
            "count": self.count,
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

    def __init__(self, db_manager: DatabaseManager, memory_manager: MemoryManager):
        """
        Args:
            db_manager    : Initialized DatabaseManager instance
            memory_manager: Initialized MemoryManager instance
        """
        self.db = db_manager
        self.memory = memory_manager

        # Session statistics
        self._session_marked_count: int = 0
        self._session_duplicate_count: int = 0
        self._session_unknown_count: int = 0

        logger.info("AttendanceRecorder initialized.")

    def process_recognition(
        self, recognition: FaceIdentityResult
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
            )

        person_id = recognition.person_id
        confidence = recognition.confidence

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
            )

        if db_action == "error":
            return AttendanceResult(
                person_id=person_id,
                confidence=confidence,
                status=STATUS_LOW_CONFIDENCE,
                count=count,
            )

        # ── STEP 13 — Update Memory File ─────────────────────────────────
        self.memory.mark_person(person_id)
        self._session_marked_count += 1

        logger.info(
            f"✅ COUNTED: {person_id} | count={count} | "
            f"confidence={confidence:.2%} | "
            f"time={now.strftime('%H:%M:%S')}"
        )

        return AttendanceResult(
            person_id=person_id,
            confidence=confidence,
            status=STATUS_MARKED,
            count=count,
        )

    def process_frame_recognitions(
        self,
        recognitions: List[FaceIdentityResult],
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
        for recognition in recognitions:
            result = self.process_recognition(recognition)
            results.append(result)

        return results

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
