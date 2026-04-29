"""
========================================================
database.py — SQLite Database Helper Module
========================================================
Handles all database operations for the attendance system.
Provides clean functions for:
    - Connecting to attendance.db
    - Inserting attendance records (with Layer 2 duplicate prevention)
    - Querying attendance history
    - Managing registered unique IDs

All functions handle their own connections and close them
cleanly — safe for long-running multi-day operation.
========================================================
"""

import logging
import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Optional

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
            self._repair_partial_migration(conn)
            self._create_current_schema(conn)
            self._migrate_schema(conn)
            self._create_current_schema(conn)
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

    def _get_tables(self, conn: sqlite3.Connection) -> set[str]:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return {row[0] for row in cursor.fetchall()}

    def _repair_partial_migration(self, conn: sqlite3.Connection):
        """Recover if a previous migration stopped after creating *_new tables."""
        cursor = conn.cursor()
        tables = self._get_tables(conn)

        if "persons" not in tables and "persons_new" in tables:
            cursor.execute("ALTER TABLE persons_new RENAME TO persons;")
            tables.remove("persons_new")
            tables.add("persons")

        if "attendance" not in tables and "attendance_new" in tables:
            cursor.execute("ALTER TABLE attendance_new RENAME TO attendance;")
            tables.remove("attendance_new")
            tables.add("attendance")

        if "persons" in tables and "persons_new" in tables:
            cursor.execute("DROP TABLE persons_new;")

        if "attendance" in tables and "attendance_new" in tables:
            cursor.execute("DROP TABLE attendance_new;")

        conn.commit()

    def _create_current_schema(self, conn: sqlite3.Connection):
        """Create the current schema if tables are missing."""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                person_id       TEXT PRIMARY KEY,
                face_signature  TEXT,
                registered_date DATE NOT NULL DEFAULT (DATE('now'))
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT    NOT NULL,
                date        DATE    NOT NULL,
                first_seen  TIME    NOT NULL,
                last_seen   TIME    NOT NULL,
                count       INTEGER NOT NULL DEFAULT 1,
                confidence  REAL,
                UNIQUE(person_id, date),
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_photos (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT    NOT NULL,
                date        DATE    NOT NULL,
                time        TIME    NOT NULL,
                count       INTEGER NOT NULL,
                confidence  REAL,
                image_path  TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (DATETIME('now')),
                UNIQUE(person_id, date, count),
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_person_date "
            "ON attendance(person_id, date);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_photos_date "
            "ON attendance_photos(date);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_photos_person_date "
            "ON attendance_photos(person_id, date);"
        )
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection):
        """Migrate older schemas to generated-face-ID attendance storage."""
        cursor = conn.cursor()
        tables = self._get_tables(conn)
        if "persons" not in tables or "attendance" not in tables:
            self._create_current_schema(conn)
            return

        cursor.execute("PRAGMA table_info(persons)")
        person_columns = {row["name"] for row in cursor.fetchall()}

        cursor.execute("PRAGMA table_info(attendance)")
        attendance_columns = {row["name"] for row in cursor.fetchall()}

        needs_migration = (
            "name" in person_columns
            or "face_signature" not in person_columns
            or "first_seen" not in attendance_columns
            or "last_seen" not in attendance_columns
            or "count" not in attendance_columns
        )
        if not needs_migration:
            return

        logger.info("Migrating database schema to generated-face-ID records")
        cursor.execute("PRAGMA foreign_keys=OFF;")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS persons_new (
                person_id       TEXT PRIMARY KEY,
                face_signature  TEXT,
                registered_date DATE NOT NULL DEFAULT (DATE('now'))
            );
            """
        )
        if "face_signature" in person_columns:
            cursor.execute(
                """
                INSERT OR IGNORE INTO persons_new
                    (person_id, face_signature, registered_date)
                SELECT person_id, face_signature, registered_date FROM persons;
                """
            )
        else:
            cursor.execute(
                """
                INSERT OR IGNORE INTO persons_new (person_id, registered_date)
                SELECT person_id, registered_date FROM persons;
                """
            )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_new (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT    NOT NULL,
                date        DATE    NOT NULL,
                first_seen  TIME    NOT NULL,
                last_seen   TIME    NOT NULL,
                count       INTEGER NOT NULL DEFAULT 1,
                confidence  REAL,
                UNIQUE(person_id, date),
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            );
            """
        )
        if "first_seen" in attendance_columns:
            cursor.execute(
                """
                INSERT OR IGNORE INTO attendance_new
                    (id, person_id, date, first_seen, last_seen, count, confidence)
                SELECT
                    id,
                    person_id,
                    date,
                    first_seen,
                    last_seen,
                    COALESCE(count, 1),
                    confidence
                FROM attendance;
                """
            )
        else:
            cursor.execute(
                """
                INSERT OR IGNORE INTO attendance_new
                    (id, person_id, date, first_seen, last_seen, count, confidence)
                SELECT id, person_id, date, time, time, 1, confidence FROM attendance;
                """
            )

        cursor.execute("DROP TABLE attendance;")
        cursor.execute("DROP TABLE persons;")
        cursor.execute("ALTER TABLE persons_new RENAME TO persons;")
        cursor.execute("ALTER TABLE attendance_new RENAME TO attendance;")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_person_date "
            "ON attendance(person_id, date);"
        )
        cursor.execute("PRAGMA foreign_keys=ON;")
        conn.commit()

    # ─────────────────────────────────────────────────────────────────────
    # ATTENDANCE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────

    def record_attendance(
        self,
        person_id: str,
        confidence: float,
        attendance_date: Optional[date] = None,
        attendance_time: Optional[datetime] = None,
        debounce_minutes: int = 5,
    ) -> tuple[str, int]:
        """
        STEP 12 — Insert or increment an attendance record.

        A person gets one row per day. The count is incremented only when
        the same generated face ID appears after the debounce window.

        Args:
            person_id   : Unique ID matching the Teachable Machine class label
            confidence  : Model confidence score (0.0 to 1.0)
            attendance_date : Date to record (defaults to today)
            attendance_time : Time to record (defaults to now)

        Returns:
            (action, count), where action is:
                "inserted"    — new person/date row
                "incremented" — count increased after debounce window
                "debounced"   — same person seen inside debounce window
                "error"       — database error
        """
        if attendance_date is None:
            attendance_date = date.today()
        if attendance_time is None:
            attendance_time = datetime.now()

        date_str = attendance_date.isoformat()
        time_str = attendance_time.strftime("%H:%M:%S")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, last_seen, count
                    FROM attendance
                    WHERE person_id=? AND date=?
                    LIMIT 1
                    """,
                    (person_id, date_str),
                )
                existing = cursor.fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO attendance
                            (person_id, date, first_seen, last_seen, count, confidence)
                        VALUES (?, ?, ?, ?, 1, ?)
                        """,
                        (
                            person_id,
                            date_str,
                            time_str,
                            time_str,
                            round(confidence, 4),
                        ),
                    )
                    conn.commit()
                    logger.info(
                        f"Attendance started: {person_id} "
                        f"on {date_str} at {time_str} | confidence={confidence:.2%}"
                    )
                    return "inserted", 1

                last_counted_at = datetime.combine(
                    attendance_date,
                    datetime.strptime(existing["last_seen"], "%H:%M:%S").time(),
                )
                if attendance_time - last_counted_at < timedelta(minutes=debounce_minutes):
                    logger.debug(
                        f"Debounced attendance count for {person_id} on {date_str}"
                    )
                    return "debounced", int(existing["count"])

                conn.execute(
                    """
                    UPDATE attendance
                    SET last_seen=?, count=count + 1, confidence=?
                    WHERE id=?
                    """,
                    (time_str, round(confidence, 4), existing["id"]),
                )
                conn.commit()
                logger.info(
                    f"Attendance count incremented: {person_id} "
                    f"on {date_str} at {time_str} | count={existing['count'] + 1}"
                )
                return "incremented", int(existing["count"]) + 1

        except sqlite3.Error as e:
            logger.error(f"Database error recording attendance for {person_id}: {e}")
            return "error", 0

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
                    """
                    SELECT 1
                    FROM attendance AS a
                    JOIN persons AS p ON p.person_id = a.person_id
                    WHERE a.person_id=? AND a.date=? AND p.face_signature IS NOT NULL
                    LIMIT 1
                    """,
                    (person_id, today_str)
                )
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking attendance for {person_id}: {e}")
            return False

    def record_attendance_photo(
        self,
        person_id: str,
        attendance_date: date,
        attendance_time: datetime,
        count: int,
        confidence: float,
        image_path: str,
    ) -> bool:
        """Store metadata for a saved attendance photo."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO attendance_photos
                        (person_id, date, time, count, confidence, image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_id, date, count) DO UPDATE SET
                        time=excluded.time,
                        confidence=excluded.confidence,
                        image_path=excluded.image_path,
                        created_at=DATETIME('now')
                    """,
                    (
                        person_id,
                        attendance_date.isoformat(),
                        attendance_time.strftime("%H:%M:%S"),
                        int(count),
                        round(confidence, 4),
                        image_path,
                    ),
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error recording attendance photo for {person_id}: {e}")
            return False

    def get_today_attendance(self) -> List[sqlite3.Row]:
        """
        Retrieve all attendance records for today.

        Returns:
            List of Row objects with columns: id, person_id, date, first_seen,
            last_seen, count, confidence
        """
        today_str = date.today().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT a.id, a.person_id, a.date, a.first_seen, a.last_seen,
                           a.count, a.confidence,
                           (
                               SELECT ap.image_path
                               FROM attendance_photos AS ap
                               WHERE ap.person_id = a.person_id
                                 AND ap.date = a.date
                               ORDER BY ap.count DESC, ap.time DESC
                               LIMIT 1
                           ) AS photo_path
                    FROM attendance AS a
                    JOIN persons AS p ON p.person_id = a.person_id
                    WHERE a.date = ? AND p.face_signature IS NOT NULL
                    ORDER BY a.first_seen ASC
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
                    SELECT a.id, a.person_id, a.date, a.first_seen, a.last_seen,
                           a.count, a.confidence,
                           (
                               SELECT ap.image_path
                               FROM attendance_photos AS ap
                               WHERE ap.person_id = a.person_id
                                 AND ap.date = a.date
                               ORDER BY ap.count DESC, ap.time DESC
                               LIMIT 1
                           ) AS photo_path
                    FROM attendance AS a
                    JOIN persons AS p ON p.person_id = a.person_id
                    WHERE a.date = ? AND p.face_signature IS NOT NULL
                    ORDER BY a.first_seen ASC
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
                    """
                    SELECT COUNT(*)
                    FROM attendance AS a
                    JOIN persons AS p ON p.person_id = a.person_id
                    WHERE a.date=? AND p.face_signature IS NOT NULL
                    """,
                    (today_str,),
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
            Row object with columns: person_id, face_signature, registered_date
            or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT person_id, face_signature, registered_date
                    FROM persons
                    WHERE person_id=?
                    """,
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
                    """
                    SELECT person_id, face_signature, registered_date
                    FROM persons
                    WHERE face_signature IS NOT NULL
                    ORDER BY person_id
                    """
                )
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching all persons: {e}")
            return []

    def add_person(self, person_id: str, face_signature: Optional[str] = None) -> bool:
        """
        Register a new unique ID in the database.

        Args:
            person_id      : Generated unique face ID
            face_signature : Signature used to match future sightings

        Returns:
            True if inserted, False if already exists or error
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO persons (person_id, face_signature)
                    VALUES (?, ?)
                    """,
                    (person_id, face_signature)
                )
                conn.commit()
                logger.info(f"Person registered: {person_id}")
                return True
        except sqlite3.IntegrityError:
            logger.debug(f"Person already exists: {person_id}")
            return False

    def update_person_signature(self, person_id: str, face_signature: str) -> bool:
        """Update the stored signature for a generated face ID."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE persons SET face_signature=? WHERE person_id=?",
                    (face_signature, person_id),
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error updating signature for {person_id}: {e}")
            return False

    def get_next_face_id(self, for_date: Optional[date] = None) -> str:
        """Return the next FACE-YYYYMMDD-NNNN ID for the given date."""
        if for_date is None:
            for_date = date.today()

        prefix = f"FACE-{for_date.strftime('%Y%m%d')}-"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT person_id FROM persons WHERE person_id LIKE ? ORDER BY person_id DESC LIMIT 1",
                    (prefix + "%",),
                )
                row = cursor.fetchone()
                if row is None:
                    return f"{prefix}0001"
                last_number = int(row["person_id"].rsplit("-", 1)[1])
                return f"{prefix}{last_number + 1:04d}"
        except (sqlite3.Error, ValueError, IndexError) as e:
            logger.error(f"Error generating next face ID: {e}")
            return f"{prefix}{int(datetime.now().timestamp())}"
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

                cursor.execute("SELECT COUNT(*) FROM persons WHERE face_signature IS NOT NULL")
                total_persons = cursor.fetchone()[0]

                today_str = date.today().isoformat()
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(a.count), 0)
                    FROM attendance AS a
                    JOIN persons AS p ON p.person_id = a.person_id
                    WHERE a.date=? AND p.face_signature IS NOT NULL
                    """,
                    (today_str,),
                )
                today_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COALESCE(SUM(a.count), 0)
                    FROM attendance AS a
                    JOIN persons AS p ON p.person_id = a.person_id
                    WHERE p.face_signature IS NOT NULL
                    """
                )
                all_time_count = cursor.fetchone()[0]

                return {
                    "total_persons": total_persons,
                    "today_count": today_count,
                    "all_time_count": all_time_count,
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_persons": 0, "today_count": 0, "all_time_count": 0}
