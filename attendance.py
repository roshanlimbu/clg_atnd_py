"""
========================================================
attendance.py — Attendance Recording Orchestrator
========================================================
STEP 11 + STEP 12 + STEP 13 + STEP 15

Coordinates the full attendance pipeline for each recognized face:

1. Layer 1 check — memory file (fast, survives restarts)
2. Layer 2 insert — SQLite database (with UNIQUE constraint)
3. Memory update  — write to today's file immediately after DB insert
4. Name lookup    — resolve person_id to display name from DB
5. Visual status  — determine color/label for live feed overlay
========================================================
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from database import DatabaseManager
from memory import MemoryManager
from recognizer import RecognitionResult
from detector import DetectedFace

logger = logging.getLogger(__name__)


# Status constants for visual overlay
STATUS_MARKED          = "marked"          # Successfully marked (green)
STATUS_ALREADY_MARKED  = "already_marked"  # Already done today (yellow)
STATUS_UNKNOWN         = "unknown"         # Unknown face (red)
STATUS_LOW_CONFIDENCE  = "low_confidence"  # Below threshold (gray)


class AttendanceResult:
    """Result of processing one detected face through the attendance pipeline."""

    def __init__(
        self,
        person_id: str,
        name: str,
        confidence: float,
        status: str,
    ):
        self.person_id = person_id
        self.name = name
        self.confidence = confidence
        self.status = status

    def as_dict(self) -> dict:
        """Convert to dict for use with detector.draw_detections()."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "confidence": self.confidence,
            "status": self.status,
        }

    def __repr__(self):
        return (
            f"AttendanceResult("
            f"person_id={self.person_id!r}, "
            f"name={self.name!r}, "
            f"confidence={self.confidence:.2%}, "
            f"status={self.status!r})"
        )


class AttendanceRecorder:
    """
    STEP 11 + 12 + 13 + 15 — Orchestrates the full attendance pipeline.

    Takes recognition results and coordinates:
        - Duplicate checking (Layer 1: memory file)
        - Database recording (Layer 2: UNIQUE constraint backup)
        - Memory file updating
        - Name resolution from database
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

        # Cache of person_id → name from database (reduces DB queries)
        self._name_cache: Dict[str, str] = {}

        # Session statistics
        self._session_marked_count: int = 0
        self._session_duplicate_count: int = 0
        self._session_unknown_count: int = 0

        # Pre-load name cache
        self._refresh_name_cache()

        logger.info("AttendanceRecorder initialized.")

    def _refresh_name_cache(self):
        """Load all registered persons into a local name cache."""
        persons = self.db.get_all_persons()
        for person in persons:
            self._name_cache[person["person_id"]] = person["name"]
        logger.debug(f"Name cache loaded: {len(self._name_cache)} persons")

    def _resolve_name(self, person_id: str) -> str:
        """
        Look up display name for a person_id.
        First checks in-memory cache, then falls back to DB.

        Args:
            person_id: Person's unique ID

        Returns:
            Display name, or the person_id itself if not found
        """
        if person_id in self._name_cache:
            return self._name_cache[person_id]

        # Not in cache — try database
        person = self.db.get_person(person_id)
        if person:
            name = person["name"]
            self._name_cache[person_id] = name
            return name

        # Not in database either — return person_id as fallback
        logger.warning(
            f"Person '{person_id}' not found in database. "
            "Add them to the persons table or retrain your model."
        )
        return person_id

    def process_recognition(
        self, recognition: RecognitionResult
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
            recognition: RecognitionResult from FaceRecognizer

        Returns:
            AttendanceResult with final status for display
        """
        # ── Handle Unknown / Low Confidence ──────────────────────────────
        if recognition.is_unknown:
            self._session_unknown_count += 1
            return AttendanceResult(
                person_id="Unknown",
                name="Unknown",
                confidence=recognition.confidence,
                status=STATUS_UNKNOWN,
            )

        if not recognition.is_above_threshold:
            return AttendanceResult(
                person_id=recognition.person_id,
                name=self._resolve_name(recognition.person_id),
                confidence=recognition.confidence,
                status=STATUS_LOW_CONFIDENCE,
            )

        person_id = recognition.person_id
        name = self._resolve_name(person_id)
        confidence = recognition.confidence

        # ── STEP 11 — Layer 1 Duplicate Check (Memory File) ──────────────
        if self.memory.is_marked_today(person_id):
            self._session_duplicate_count += 1
            logger.debug(
                f"Layer 1 duplicate blocked: {person_id} ({name}) "
                "— already in today's memory file"
            )
            return AttendanceResult(
                person_id=person_id,
                name=name,
                confidence=confidence,
                status=STATUS_ALREADY_MARKED,
            )

        # ── STEP 12 — Record in SQLite Database ──────────────────────────
        now = datetime.now()
        db_success = self.db.record_attendance(
            person_id=person_id,
            name=name,
            confidence=confidence,
            attendance_date=now.date(),
            attendance_time=now,
        )

        if not db_success:
            # Layer 2 caught a duplicate (edge case)
            logger.info(
                f"Layer 2 duplicate blocked: {person_id} ({name}) "
                "— UNIQUE constraint in database"
            )
            # Still update memory file to sync with DB state
            self.memory.mark_person(person_id)
            return AttendanceResult(
                person_id=person_id,
                name=name,
                confidence=confidence,
                status=STATUS_ALREADY_MARKED,
            )

        # ── STEP 13 — Update Memory File ─────────────────────────────────
        self.memory.mark_person(person_id)
        self._session_marked_count += 1

        logger.info(
            f"✅ MARKED: {person_id} ({name}) | "
            f"confidence={confidence:.2%} | "
            f"time={now.strftime('%H:%M:%S')}"
        )

        return AttendanceResult(
            person_id=person_id,
            name=name,
            confidence=confidence,
            status=STATUS_MARKED,
        )

    def process_frame_recognitions(
        self,
        recognitions: List[RecognitionResult],
    ) -> List[AttendanceResult]:
        """
        Process all recognition results from a single frame.

        Args:
            recognitions: List of RecognitionResult objects (one per detected face)

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
        logger.info(f"  Total marked today   : {stats['today_total']}")
        logger.info(f"  Registered persons   : {stats['registered_persons']}")
        logger.info(f"  Session marked       : {stats['session_marked']}")
        logger.info(f"  Duplicates blocked   : {stats['session_duplicates_blocked']}")
        logger.info(f"  Unknown faces seen   : {stats['session_unknowns']}")
        logger.info("-" * 55)

        if records:
            logger.info("  Attendance log:")
            for rec in records:
                logger.info(
                    f"    {rec['time']}  {rec['person_id']:<12}  "
                    f"{rec['name']:<25}  {rec['confidence']:.0%}"
                )
        else:
            logger.info("  No attendance records yet today.")

        logger.info("=" * 55)