"""
========================================================
STEP 3 + STEP 4 — Project Setup & Database Initialization
========================================================
Run this ONCE after converting the model.
Sets up folder structure, installs dependencies,
and initializes the SQLite attendance database.

Usage:
    python setup.py

What it does:
    - Creates all required directories
    - Checks all package installations
    - Creates attendance.db with generated face IDs + attendance counts
    - Adds the UNIQUE constraint for one row per face per day
========================================================
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path


# ── Required packages ────────────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    "opencv-python",
    "ultralytics",
    "mediapipe",
    "tensorflow",
    "tensorflowjs",
    "numpy",
    "Pillow",
]

# Map pip package names → import names (they differ sometimes)
IMPORT_MAP = {
    "opencv-python": "cv2",
    "ultralytics": "ultralytics",
    "mediapipe": "mediapipe",
    "tensorflow": "tensorflow",
    "tensorflowjs": "tensorflowjs",
    "numpy": "numpy",
    "Pillow": "PIL",
}


def check_python_version():
    """Ensure a Python version supported by the ML dependencies is being used."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"[ERROR] Python 3.8+ required. You have {major}.{minor}")
        sys.exit(1)
    if major > 3 or (major == 3 and minor >= 13):
        print(f"[ERROR] Python {major}.{minor} is too new for TensorFlow/MediaPipe.")
        print("        Use Python 3.12 for this project.")
        sys.exit(1)
    print(f"[OK] Python {major}.{minor} — version compatible.")


def install_packages():
    """Install all required packages via pip if not already installed."""
    print("\n[INFO] Checking and installing required packages...")
    for pkg in REQUIRED_PACKAGES:
        import_name = IMPORT_MAP[pkg]
        try:
            __import__(import_name)
            print(f"  [OK] {pkg} — already installed")
        except ImportError:
            print(f"  [INSTALLING] {pkg}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  [OK] {pkg} — installed successfully")
            else:
                print(f"  [ERROR] Failed to install {pkg}")
                print(f"          {result.stderr[-300:]}")


def create_directory_structure(base_dir: Path):
    """Create all required project directories."""
    print("\n[INFO] Creating project directory structure...")

    directories = [
        base_dir / "models",
        base_dir / "attendance_memory",
        base_dir / "attendance_photos",
        base_dir / "converted_model",
        base_dir / "logs",
    ]

    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d.relative_to(base_dir)}/")

    print("[OK] All directories ready.")


def initialize_database(base_dir: Path):
    """
    STEP 4 — Create SQLite attendance.db with all required tables.

    Tables:
        persons    — stores registered unique IDs
        attendance — stores daily attendance counts with UNIQUE(person_id, date)
    """
    db_path = base_dir / "attendance.db"
    print(f"\n[INFO] Initializing database at: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # ── Enable WAL mode for better concurrent read performance ────────────
    cursor.execute("PRAGMA journal_mode=WAL;")

    # ── persons table ─────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id       TEXT PRIMARY KEY,
            face_signature  TEXT,
            registered_date DATE NOT NULL DEFAULT (DATE('now'))
        );
    """)
    print("  [OK] Table 'persons' created (or already exists).")

    # ── attendance table with Layer 2 duplicate prevention ────────────────
    cursor.execute("""
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
    """)
    print("  [OK] Table 'attendance' created with UNIQUE(person_id, date) constraint.")

    cursor.execute("""
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
    """)
    print("  [OK] Table 'attendance_photos' created for saved face photos.")

    # ── Index for fast date-range queries ─────────────────────────────────
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_date
        ON attendance(date);
    """)
    print("  [OK] Index on attendance(date) created.")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_person_date
        ON attendance(person_id, date);
    """)
    print("  [OK] Index on attendance(person_id, date) created.")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_photos_date
        ON attendance_photos(date);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_photos_person_date
        ON attendance_photos(person_id, date);
    """)
    print("  [OK] Indexes on attendance_photos created.")

    conn.commit()
    migrate_database_schema(conn)

    # ── Verify schema ──────────────────────────────────────────────────────
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n[OK] Database tables confirmed: {tables}")

    conn.close()
    print(f"[OK] Database initialized: {db_path}")
    return db_path


def migrate_database_schema(conn: sqlite3.Connection):
    """Migrate old schemas to generated-face-ID attendance counts."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
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

    print("  [MIGRATE] Updating database for generated face IDs and counts.")
    cursor.execute("PRAGMA foreign_keys=OFF;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons_new (
            person_id       TEXT PRIMARY KEY,
            face_signature  TEXT,
            registered_date DATE NOT NULL DEFAULT (DATE('now'))
        );
    """)
    if "face_signature" in person_columns:
        cursor.execute("""
            INSERT OR IGNORE INTO persons_new
                (person_id, face_signature, registered_date)
            SELECT person_id, face_signature, registered_date FROM persons;
        """)
    else:
        cursor.execute("""
            INSERT OR IGNORE INTO persons_new (person_id, registered_date)
            SELECT person_id, registered_date FROM persons;
        """)
    cursor.execute("""
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
    """)
    if "first_seen" in attendance_columns:
        cursor.execute("""
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
        """)
    else:
        cursor.execute("""
            INSERT OR IGNORE INTO attendance_new
                (id, person_id, date, first_seen, last_seen, count, confidence)
            SELECT id, person_id, date, time, time, 1, confidence FROM attendance;
        """)
    cursor.execute("DROP TABLE attendance;")
    cursor.execute("DROP TABLE persons;")
    cursor.execute("ALTER TABLE persons_new RENAME TO persons;")
    cursor.execute("ALTER TABLE attendance_new RENAME TO attendance;")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_person_date
        ON attendance(person_id, date);
    """)
    cursor.execute("PRAGMA foreign_keys=ON;")
    conn.commit()


def prune_placeholder_persons(db_path: Path):
    """Remove old fixed-label placeholder rows from the classifier flow."""
    print("\n[INFO] Pruning old fixed-label placeholder IDs...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM attendance
        WHERE person_id IN (
            SELECT person_id FROM persons WHERE face_signature IS NULL
        )
    """)
    removed_attendance = cursor.rowcount
    cursor.execute("""
        DELETE FROM persons
        WHERE face_signature IS NULL
    """)
    print(
        f"  [OK] Removed {cursor.rowcount} placeholder IDs "
        f"and {removed_attendance} legacy attendance rows"
    )
    conn.commit()
    conn.close()


def verify_model_files(base_dir: Path):
    """Check legacy Teachable Machine files, if you still use conversion tools."""
    print("\n[INFO] Checking optional Teachable Machine model files...")
    models_dir = base_dir / "models"
    required = ["model.json", "weights.bin", "metadata.json"]
    all_present = True

    for f in required:
        path = models_dir / f
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK] {f} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {f} — only needed for convert_model.py")
            all_present = False

    if not all_present:
        print("\n[INFO] Model files are optional for generated face IDs.")
    else:
        # Check if already converted
        h5_path = base_dir / "converted_model" / "model.h5"
        if h5_path.exists():
            print(f"\n[OK] Converted model already exists: {h5_path}")
        else:
            print(f"\n[INFO] Run 'python convert_model.py' to convert the model next.")


def print_project_summary(base_dir: Path):
    """Print a summary of the project structure and next steps."""
    print("\n" + "=" * 60)
    print("  PROJECT SETUP COMPLETE")
    print("=" * 60)
    print(f"\n  Project directory: {base_dir}")
    print("""
  Directory structure:
  ├── models/
  │   ├── model.json          ← Optional Teachable Machine export
  │   ├── weights.bin         ← Optional Teachable Machine export
  │   └── metadata.json       ← Optional Teachable Machine export
  ├── attendance_memory/      ← Daily memory text files (auto-created)
  ├── attendance_photos/      ← Saved face photos for each counted arrival
  ├── converted_model/        ← Optional legacy classifier output
  ├── logs/                   ← Runtime logs
  ├── attendance.db           ← SQLite database ✓
  ├── convert_model.py        ← Step 2: run once
  ├── setup.py                ← Step 3+4: this file
  ├── database.py             ← DB helpers
  ├── memory.py               ← Memory file helpers
  ├── detector.py             ← Face detection (YOLO + MediaPipe)
  ├── face_identity.py        ← Generated face ID matching
  ├── recognizer.py           ← Optional legacy classifier
  ├── camera.py               ← Camera feed handler
  ├── attendance.py           ← Attendance recording logic
  └── main.py                 ← Main entry point

  NEXT STEPS:
  1. Grant camera permission to your terminal app
  2. Run: python main.py
""")


def main():
    print("=" * 60)
    print("  Attendance System — Project Setup")
    print("=" * 60)

    base_dir = Path(__file__).parent

    check_python_version()
    create_directory_structure(base_dir)
    install_packages()

    db_path = initialize_database(base_dir)
    prune_placeholder_persons(db_path)
    verify_model_files(base_dir)
    print_project_summary(base_dir)


if __name__ == "__main__":
    main()
