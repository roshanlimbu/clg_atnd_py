"""
========================================================
memory.py — Daily Memory File Manager (Layer 1 Duplicate Prevention)
========================================================
STEP 5 (partial), STEP 11, STEP 13, STEP 15

The memory file is a simple plain-text file that records
which persons have already been marked present today.

One file per day:  attendance_memory/memory_YYYY-MM-DD.txt
Each line in the file is one person_id.

Why a file instead of RAM?
    - Survives program crashes and restarts
    - Zero memory dependency
    - Readable/auditable by humans
    - Works across multiple-day continuous runs

Two-layer duplicate prevention:
    Layer 1 (this file) — fast in-memory set loaded from file at startup
    Layer 2 (database)  — UNIQUE constraint as safety net

Example file content:
    STU001
    STU003
    STU007
========================================================
"""

import os
import logging
from pathlib import Path
from datetime import date
from typing import Set

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages the daily memory file for tracking who has been
    marked present today — Layer 1 duplicate prevention.

    At program startup: loads today's file (if it exists) into a
    fast in-memory set for O(1) lookups during live recognition.

    When a new person is marked: writes their ID to the file
    immediately and adds to the in-memory set.
    """

    def __init__(self, memory_dir: Path):
        """
        Initialize memory manager for today's date.

        Args:
            memory_dir: Path to the attendance_memory/ directory
        """
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.today = date.today()
        self.memory_file_path = self._get_memory_file_path(self.today)

        # In-memory set for O(1) duplicate checks during live feed
        self._marked_today: Set[str] = set()

        # Load existing memory on startup (handles restarts mid-day)
        self._load_memory_file()

    def _get_memory_file_path(self, for_date: date) -> Path:
        """Build the full path for a date's memory file."""
        filename = f"memory_{for_date.isoformat()}.txt"
        return self.memory_dir / filename

    def _load_memory_file(self):
        """
        STEP 5 — Load today's memory file into the in-memory set.

        If the file exists (program restarted mid-day):
            → Read all person_ids into the set
            → These persons are already marked, won't be double-marked

        If the file doesn't exist (new day or first run):
            → Create an empty file
            → In-memory set stays empty — everyone starts fresh
        """
        if self.memory_file_path.exists():
            # File exists — program restarted, load who was already marked
            with open(self.memory_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    person_id = line.strip()
                    if person_id:  # Skip empty lines
                        self._marked_today.add(person_id)

            logger.info(
                f"Memory loaded from {self.memory_file_path.name}: "
                f"{len(self._marked_today)} persons already marked today"
            )
            if self._marked_today:
                logger.debug(f"Already marked today: {self._marked_today}")
        else:
            # New day or first run — create empty memory file
            with open(self.memory_file_path, "w", encoding="utf-8") as f:
                pass  # Create empty file

            logger.info(
                f"New memory file created: {self.memory_file_path.name} "
                "(fresh start — no one marked yet today)"
            )

    def is_marked_today(self, person_id: str) -> bool:
        """
        STEP 11 — Layer 1 duplicate check.

        Check in-memory set (loaded from today's file at startup).
        This is the PRIMARY and FASTEST check — runs on every recognized face.

        Args:
            person_id: Person's unique ID (from Teachable Machine class label)

        Returns:
            True  — already marked today (ignore this face)
            False — not marked yet (proceed to record attendance)
        """
        return person_id in self._marked_today

    def mark_person(self, person_id: str) -> bool:
        """
        STEP 13 — Record a person as marked in memory file and in-memory set.

        Called immediately AFTER a successful database insert.
        Writes person_id to today's file on a new line.
        Also adds to in-memory set for instant future lookups.

        If program crashes after DB insert but before this write,
        the in-memory set won't have this person — but on restart,
        the DB UNIQUE constraint (Layer 2) will still block duplicates.

        Args:
            person_id: Person's unique ID to mark

        Returns:
            True  — successfully written to file
            False — file write error (non-critical, DB is the backup)
        """
        try:
            with open(self.memory_file_path, "a", encoding="utf-8") as f:
                f.write(person_id + "\n")

            self._marked_today.add(person_id)

            logger.debug(
                f"Memory updated: {person_id} written to {self.memory_file_path.name}"
            )
            return True

        except OSError as e:
            logger.error(
                f"Failed to write to memory file for {person_id}: {e}. "
                "Continuing — Layer 2 DB constraint will still prevent duplicates."
            )
            # Still add to in-memory set so this session doesn't double-mark
            self._marked_today.add(person_id)
            return False

    def check_and_refresh_date(self) -> bool:
        """
        STEP 15 — Check if the date has changed (midnight rollover).

        Called periodically during long-running operation.
        If today's date is different from the memory manager's date,
        it means midnight has passed — start a fresh memory for the new day.

        Returns:
            True  — date changed, memory reset for new day
            False — same day, no change
        """
        current_date = date.today()

        if current_date != self.today:
            old_date = self.today
            self.today = current_date
            self.memory_file_path = self._get_memory_file_path(self.today)
            self._marked_today.clear()

            # Create fresh empty file for the new day
            with open(self.memory_file_path, "w", encoding="utf-8") as f:
                pass

            logger.info(
                f"Midnight rollover detected: {old_date} → {self.today}. "
                f"New memory file: {self.memory_file_path.name}. "
                "Attendance reset — everyone can be marked again."
            )
            return True

        return False

    def get_marked_count(self) -> int:
        """Return how many persons are marked present today."""
        return len(self._marked_today)

    def get_marked_persons(self) -> Set[str]:
        """Return a copy of the set of person_ids marked today."""
        return self._marked_today.copy()

    def get_memory_file_list(self) -> list:
        """
        Return a sorted list of all memory files (for audit purposes).
        Each file represents one day's attendance history.
        """
        files = sorted(self.memory_dir.glob("memory_*.txt"))
        return [
            {
                "date": f.stem.replace("memory_", ""),
                "path": f,
                "size_bytes": f.stat().st_size,
            }
            for f in files
        ]

    def get_status_summary(self) -> dict:
        """Return current memory status for display/logging."""
        return {
            "today": self.today.isoformat(),
            "memory_file": self.memory_file_path.name,
            "marked_count": len(self._marked_today),
            "marked_persons": list(self._marked_today),
            "file_exists": self.memory_file_path.exists(),
        }