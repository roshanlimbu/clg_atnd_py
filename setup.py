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
    - Creates attendance.db with persons + attendance tables
    - Adds the UNIQUE constraint for duplicate prevention
    - Inserts a sample person record for testing
========================================================
"""

import os
import sys
import json
import sqlite3
import subprocess
from pathlib import Path
from datetime import date


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
    """Ensure Python 3.8+ is being used."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"[ERROR] Python 3.8+ required. You have {major}.{minor}")
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
        persons    — stores registered people (ID, name, registration date)
        attendance — stores daily attendance records with UNIQUE(person_id, date)
                     The UNIQUE constraint is Layer 2 duplicate prevention.
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
            name            TEXT NOT NULL,
            registered_date DATE NOT NULL DEFAULT (DATE('now'))
        );
    """)
    print("  [OK] Table 'persons' created (or already exists).")

    # ── attendance table with Layer 2 duplicate prevention ────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   TEXT    NOT NULL,
            name        TEXT    NOT NULL,
            date        DATE    NOT NULL,
            time        TIME    NOT NULL,
            confidence  REAL,
            UNIQUE(person_id, date),
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );
    """)
    print("  [OK] Table 'attendance' created with UNIQUE(person_id, date) constraint.")

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

    conn.commit()

    # ── Verify schema ──────────────────────────────────────────────────────
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n[OK] Database tables confirmed: {tables}")

    conn.close()
    print(f"[OK] Database initialized: {db_path}")
    return db_path


def add_sample_persons(db_path: Path):
    """
    Insert sample/placeholder persons into the persons table.
    Replace these with your actual registered student IDs and names.
    The class labels in Teachable Machine metadata.json must match person_id exactly.
    """
    sample_persons = [
        # (person_id,  name)
        # person_id must EXACTLY match the class label in your Teachable Machine model
        ("Unknown",    "Unknown Person"),   # Required — handles unrecognized faces
        # Add your actual registered persons below:
        # ("STU001", "Alice Johnson"),
        # ("STU002", "Bob Smith"),
        # ("STU003", "Carol Davis"),
    ]

    print("\n[INFO] Inserting sample persons into database...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for person_id, name in sample_persons:
        cursor.execute("""
            INSERT OR IGNORE INTO persons (person_id, name, registered_date)
            VALUES (?, ?, DATE('now'))
        """, (person_id, name))
        if cursor.rowcount > 0:
            print(f"  [OK] Inserted: {person_id} — {name}")
        else:
            print(f"  [SKIP] Already exists: {person_id} — {name}")

    conn.commit()
    conn.close()


def verify_model_files(base_dir: Path):
    """Check if Teachable Machine model files are present."""
    print("\n[INFO] Checking Teachable Machine model files...")
    models_dir = base_dir / "models"
    required = ["model.json", "weights.bin", "metadata.json"]
    all_present = True

    for f in required:
        path = models_dir / f
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK] {f} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {f} — place this file in models/ directory")
            all_present = False

    if not all_present:
        print("\n[WARNING] Some model files are missing.")
        print("          Export your Teachable Machine model and place files in models/")
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
  │   ├── model.json          ← Teachable Machine export
  │   ├── weights.bin         ← Teachable Machine export
  │   └── metadata.json       ← Teachable Machine export
  ├── attendance_memory/      ← Daily memory text files (auto-created)
  ├── converted_model/
  │   └── model.h5            ← Generated by convert_model.py
  ├── logs/                   ← Runtime logs
  ├── attendance.db           ← SQLite database ✓
  ├── convert_model.py        ← Step 2: run once
  ├── setup.py                ← Step 3+4: this file
  ├── database.py             ← DB helpers
  ├── memory.py               ← Memory file helpers
  ├── detector.py             ← Face detection (YOLO + MediaPipe)
  ├── recognizer.py           ← Face recognition (Keras model)
  ├── camera.py               ← Camera feed handler
  ├── attendance.py           ← Attendance recording logic
  └── main.py                 ← Main entry point

  NEXT STEPS:
  1. Place Teachable Machine model files in models/
  2. Run: python convert_model.py
  3. Add your registered persons to persons table in attendance.db
  4. Run: python main.py
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
    add_sample_persons(db_path)
    verify_model_files(base_dir)
    print_project_summary(base_dir)


if __name__ == "__main__":
    main()