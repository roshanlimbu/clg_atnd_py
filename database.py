"""
========================================================
database.py — SQLite Database Helper Module
========================================================
Handles all database operations for the attendance system.
Provides clean functions for:
    - Connecting to attendance.db
    - Inserting attendance records (with Layer 2 duplicate prevention)
    - Querying attendance history
    - Managing person records

All functions handle their own connections and close them
cleanly — safe for long-running multi-day operation.
========================================================
"""

import sqlite3
import logging
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all SQLite operations for the attendance system.
    Uses context managers to ensure connections are always closed,
    even during crashes or errors.
    """

    def __init__(self, db_path: Path):
        """
        Initialize with path to attendance.db.

        Args:
            db_path: Full path to the attendance.db file
        """
        self.db_path = db_path
        self._verify_database()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Create and return a new database connection.
        Called fresh for each operation — ensures no stale connections
        during multi-day continuous runs.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10,          # Wait up to 10s if DB is locked
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row   # Enables column name access
        conn.execute("PRAGMA journal_mode=WAL;")   # Better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON;")    # Enforce FK constraints
        return conn

    def _verify_database(self):
        """Verify the database file exists and has the required tables."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Run setup.py first to initialize the database."
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            tables = {row[0] for row in cursor.fetchall()}
            required = {"persons", "attendance"}
            missing = required - tables
            if missing:
                raise RuntimeError(
                    f"Missing tables in database: {missing}. "
                    "Run setup.py to reinitialize."
                )

        logger.info(f"Database verified at: {self.db_path}")

    # ─────────────────────────────────────────────────────────────────────
    # ATTENDANCE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────

    def record_attendance(
        self,
        person_id: str,
        name: str,
        confidence: float,
        attendance_date: Optional[date] = None,
        attendance_time: Optional[datetime] = None,
    ) -> bool:
        """
        STEP 12 — Insert an attendance record into the database.

        Layer 2 duplicate prevention: The UNIQUE(person_id, date) constraint
        will cause an IntegrityError if a duplicate is attempted.
        This function catches that silently and returns False.

        Args:
            person_id   : Unique ID matching the Teachable Machine class label
            name        : Human-readable display name
            confidence  : Model confidence score (0.0 to 1.0)
            attendance_date : Date to record (defaults to today)
            attendance_time : Time to record (defaults to now)

        Returns:
            True  — if record was successfully inserted
            False — if duplicate (already marked today) or other DB error
        """
        if attendance_date is None:
            attendance_date = date.today()
        if attendance_time is None:
            attendance_time = datetime.now()

        date_str = attendance_date.isoformat()
        time_str = attendance_time.strftime("%H:%M:%S")

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO attendance (person_id, name, date, time, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (person_id, name, date_str, time_str, round(confidence, 4))
                )
                conn.commit()
                logger.info(
                    f"Attendance recorded: {person_id} ({name}) "
                    f"on {date_str} at {time_str} | confidence={confidence:.2%}"
                )
                return True

        except sqlite3.IntegrityError:
            # UNIQUE constraint violation — person already marked today
            # This is Layer 2 safety net catching edge cases
            logger.debug(
                f"Layer 2 duplicate blocked: {person_id} on {date_str} "
                "(UNIQUE constraint — already in DB)"
            )
            return False

        except sqlite3.Error as e:
            logger.error(f"Database error recording attendance for {person_id}: {e}")
            return False

    def is_marked_today(self, person_id: str) -> bool:
        """
        Check if a person is already marked present today (database query).
        Note: The primary duplicate check uses the memory file (Layer 1).
        This is only used for verification or UI purposes.

        Args:
            person_id: Person's unique ID

        Returns:
            True if already marked today, False otherwise
        """
        today_str = date.today().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM attendance WHERE person_id=? AND date=? LIMIT 1",
                    (person_id, today_str)
                )
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking attendance for {person_id}: {e}")
            return False

    def get_today_attendance(self) -> List[sqlite3.Row]:
        """
        Retrieve all attendance records for today.

        Returns:
            List of Row objects with columns: id, person_id, name, date, time, confidence
        """
        today_str = date.today().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, person_id, name, date, time, confidence
                    FROM attendance
                    WHERE date = ?
                    ORDER BY time ASC
                    """,
                    (today_str,)
                )
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching today's attendance: {e}")
            return []

    def get_attendance_by_date(self, query_date: date) -> List[sqlite3.Row]:
        """
        Retrieve all attendance records for a specific date.

        Args:
            query_date: The date to query

        Returns:
            List of attendance Row objects
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, person_id, name, date, time, confidence
                    FROM attendance
                    WHERE date = ?
                    ORDER BY time ASC
                    """,
                    (query_date.isoformat(),)
                )
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching attendance for {query_date}: {e}")
            return []

    def get_attendance_count_today(self) -> int:
        """Return count of unique persons marked present today."""
        today_str = date.today().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM attendance WHERE date=?",
                    (today_str,)
                )
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Error counting today attendance: {e}")
            return 0

    # ─────────────────────────────────────────────────────────────────────
    # PERSONS OPERATIONS
    # ─────────────────────────────────────────────────────────────────────

    def get_person(self, person_id: str) -> Optional[sqlite3.Row]:
        """
        Look up a person by their ID.

        Args:
            person_id: Unique person identifier

        Returns:
            Row object with columns: person_id, name, registered_date
            or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT person_id, name, registered_date FROM persons WHERE person_id=?",
                    (person_id,)
                )
                return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Error fetching person {person_id}: {e}")
            return None

    def get_all_persons(self) -> List[sqlite3.Row]:
        """Return all registered persons."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT person_id, name, registered_date FROM persons ORDER BY name"
                )
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching all persons: {e}")
            return []

    def add_person(self, person_id: str, name: str) -> bool:
        """
        Register a new person in the database.

        Args:
            person_id : Must exactly match the class label in Teachable Machine
            name      : Human-readable display name

        Returns:
            True if inserted, False if already exists or error
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO persons (person_id, name) VALUES (?, ?)",
                    (person_id, name)
                )
                conn.commit()
                logger.info(f"Person registered: {person_id} — {name}")
                return True
        except sqlite3.IntegrityError:
            logger.debug(f"Person already exists: {person_id}")
            return False
        except sqlite3.Error as e:
            logger.error(f"Error adding person {person_id}: {e}")
            return False

    def get_statistics(self) -> dict:
        """
        Return summary statistics for display/logging.

        Returns:
            Dictionary with total_persons, today_count, all_time_count
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM persons WHERE person_id != 'Unknown'")
                total_persons = cursor.fetchone()[0]

                today_str = date.today().isoformat()
                cursor.execute(
                    "SELECT COUNT(*) FROM attendance WHERE date=?", (today_str,)
                )
                today_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM attendance")
                all_time_count = cursor.fetchone()[0]

                return {
                    "total_persons": total_persons,
                    "today_count": today_count,
                    "all_time_count": all_time_count,
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_persons": 0, "today_count": 0, "all_time_count": 0}