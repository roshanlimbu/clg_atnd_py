"""
Reset database: remove all old guest registrations and attendance data.
Keeps internal team members intact.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "attendance.db"

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Show what we're keeping vs removing
cur.execute("SELECT COUNT(*) as c FROM persons WHERE role = 'internal'")
internal_count = cur.fetchone()["c"]

cur.execute("SELECT COUNT(*) as c FROM persons WHERE role != 'internal'")
guest_count = cur.fetchone()["c"]

cur.execute("SELECT COUNT(*) as c FROM attendance")
attendance_count = cur.fetchone()["c"]

cur.execute("SELECT COUNT(*) as c FROM attendance_photos")
photos_count = cur.fetchone()["c"]

print(f"\n=== Current State ===")
print(f"  Internal team members (KEEPING): {internal_count}")
print(f"  Guest/other persons (REMOVING):  {guest_count}")
print(f"  Attendance records (REMOVING):   {attendance_count}")
print(f"  Attendance photos (REMOVING):    {photos_count}")

# Check for pending_faces table
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pending_faces'")
has_pending = cur.fetchone() is not None
pending_count = 0
if has_pending:
    cur.execute("SELECT COUNT(*) as c FROM pending_faces")
    pending_count = cur.fetchone()["c"]
    print(f"  Pending faces (REMOVING):        {pending_count}")

# Check for auto_registered_visitors table
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auto_registered_visitors'")
has_auto = cur.fetchone() is not None
auto_count = 0
if has_auto:
    cur.execute("SELECT COUNT(*) as c FROM auto_registered_visitors")
    auto_count = cur.fetchone()["c"]
    print(f"  Auto-registered (REMOVING):      {auto_count}")

print(f"\n--- Proceeding with cleanup ---")

# Disable FK constraints during cleanup
cur.execute("PRAGMA foreign_keys=OFF;")

# 1. Remove attendance photos for guests
cur.execute("""
    DELETE FROM attendance_photos 
    WHERE person_id IN (SELECT person_id FROM persons WHERE role != 'internal')
""")
print(f"  Deleted {cur.rowcount} attendance photos")

# 2. Remove attendance records for guests
cur.execute("""
    DELETE FROM attendance 
    WHERE person_id IN (SELECT person_id FROM persons WHERE role != 'internal')
""")
print(f"  Deleted {cur.rowcount} attendance records")

# 3. Remove guest persons
cur.execute("DELETE FROM persons WHERE role != 'internal'")
print(f"  Deleted {cur.rowcount} guest persons")

# 4. Clear pending faces
if has_pending:
    cur.execute("DELETE FROM pending_faces")
    print(f"  Cleared {pending_count} pending faces")

# 5. Clear auto-registered visitors
if has_auto:
    cur.execute("DELETE FROM auto_registered_visitors")
    print(f"  Cleared {auto_count} auto-registered entries")

cur.execute("PRAGMA foreign_keys=ON;")
conn.commit()

# Verify
cur.execute("SELECT person_id, display_name, role FROM persons ORDER BY person_id")
remaining = cur.fetchall()
print(f"\n=== Remaining Persons: {len(remaining)} ===")
for p in remaining:
    print(f"  {p['person_id']:30s}  role: {p['role']:10s}  name: {p['display_name']}")

# Vacuum to reclaim space
conn.execute("VACUUM")
print(f"\n=== Database vacuumed. Done! ===")
print("Restart main.py to load the clean state.")

conn.close()
