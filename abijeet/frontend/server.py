"""
Local dashboard server for the face attendance system.

Run from the project root with:
    ../venv312/bin/python frontend/server.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FRONTEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
DATABASE_PATH = PROJECT_DIR / "attendance.db"
INTERNAL_PHOTOS_DIR = PROJECT_DIR / "internal_team_photos"
HOST = "127.0.0.1"
PORT = 8000
ACTIVE_WINDOW_MINUTES = 5
FACE_ROW_LIMIT = 100
PORT_SCAN_LIMIT = 20


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve static dashboard files and JSON API responses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            self._send_json(get_dashboard_summary())
            return

        if parsed.path == "/api/health":
            self._send_json({"ok": True, "database": str(DATABASE_PATH)})
            return

        if parsed.path == "/api/internal-team":
            self._send_json({"ok": True, "members": get_internal_team()})
            return

        if (
            parsed.path.startswith("/attendance_photos/")
            or parsed.path.startswith("/internal_team_photos/")
        ):
            self._send_managed_image(parsed.path)
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/internal-team":
            self._handle_internal_team_post()
            return

        self.send_error(404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_managed_image(self, request_path: str):
        relative_path = unquote(request_path).lstrip("/")
        roots = [
            (PROJECT_DIR / "attendance_photos").resolve(),
            INTERNAL_PHOTOS_DIR.resolve(),
        ]
        image_path = (PROJECT_DIR / relative_path).resolve()

        if not any(image_path == root or root in image_path.parents for root in roots):
            self.send_error(404)
            return

        if not image_path.is_file():
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(image_path)))
        self.send_header("Content-Length", str(image_path.stat().st_size))
        self.end_headers()
        with image_path.open("rb") as file_obj:
            self.copyfile(file_obj, self.wfile)

    def _handle_internal_team_post(self):
        try:
            result = create_internal_team_member(self)
            status = 201 if result.get("ok") else 400
            self._send_json(result, status=status)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)


def get_connection() -> sqlite3.Connection:
    """Open a short-lived read-only SQLite connection."""
    uri = f"file:{DATABASE_PATH}?mode=ro&cache=shared"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_database_schema():
    """Run lightweight migrations before dashboard reads/writes."""
    from database import DatabaseManager

    DatabaseManager(DATABASE_PATH)


def get_dashboard_summary() -> dict:
    """Read the latest attendance state for the dashboard."""
    ensure_database_schema()
    today = date.today().isoformat()
    now = datetime.now()
    active_cutoff = (now - timedelta(minutes=ACTIVE_WINDOW_MINUTES)).strftime("%H:%M:%S")

    try:
        with get_connection() as conn:
            has_photos = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'attendance_photos'
                """
            ).fetchone() is not None

            photo_select = (
                """
                (
                    SELECT ap.image_path
                    FROM attendance_photos AS ap
                    WHERE ap.person_id = a.person_id
                      AND ap.date = a.date
                    ORDER BY ap.count DESC, ap.time DESC
                    LIMIT 1
                ) AS photo_path
                """
                if has_photos
                else "NULL AS photo_path"
            )

            totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    COUNT(*) AS unique_faces,
                    COALESCE(SUM(
                        CASE WHEN a.last_seen >= ? THEN 1 ELSE 0 END
                    ), 0) AS active_faces
                FROM attendance AS a
                JOIN persons AS p ON p.person_id = a.person_id
                WHERE a.date = ? AND p.face_signature IS NOT NULL
                  AND COALESCE(p.role, 'guest') != 'internal'
                """,
                (active_cutoff, today),
            ).fetchone()

            rows = conn.execute(
                f"""
                SELECT
                    a.person_id,
                    a.date,
                    a.first_seen,
                    a.last_seen,
                    a.count,
                    a.confidence,
                    p.display_name,
                    p.role,
                    {photo_select}
                FROM attendance AS a
                JOIN persons AS p ON p.person_id = a.person_id
                WHERE a.date = ? AND p.face_signature IS NOT NULL
                  AND COALESCE(p.role, 'guest') != 'internal'
                ORDER BY a.last_seen DESC, a.first_seen DESC
                LIMIT ?
                """,
                (today, FACE_ROW_LIMIT),
            ).fetchall()

            return {
                "ok": True,
                "date": today,
                "server_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "poll_seconds": 10,
                "active_window_minutes": ACTIVE_WINDOW_MINUTES,
                "row_limit": FACE_ROW_LIMIT,
                "total_count": int(totals["total_count"]),
                "unique_faces": int(totals["unique_faces"]),
                "active_faces": int(totals["active_faces"]),
                "faces": [serialize_face(row) for row in rows],
                "internal_team": get_internal_team(conn),
            }
    except sqlite3.Error as exc:
        return {
            "ok": False,
            "error": str(exc),
            "date": today,
            "server_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "poll_seconds": 10,
            "active_window_minutes": ACTIVE_WINDOW_MINUTES,
            "row_limit": FACE_ROW_LIMIT,
            "total_count": 0,
            "unique_faces": 0,
            "active_faces": 0,
            "faces": [],
            "internal_team": [],
        }


def serialize_face(row: sqlite3.Row) -> dict:
    """Convert a SQLite row to JSON-safe dashboard data."""
    confidence = row["confidence"]
    photo_path = row["photo_path"]
    return {
        "person_id": row["person_id"],
        "date": row["date"],
        "display_name": row["display_name"],
        "role": row["role"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "count": int(row["count"]),
        "confidence": None if confidence is None else round(float(confidence), 4),
        "photo_url": None if photo_path is None else f"/{photo_path}",
    }


def get_internal_team(conn: sqlite3.Connection | None = None) -> list[dict]:
    """Return saved internal team members for the dashboard."""
    close_conn = False
    if conn is None:
        ensure_database_schema()
        conn = get_connection()
        close_conn = True

    try:
        rows = conn.execute(
            """
            SELECT person_id, display_name, reference_photo_path, registered_date
            FROM persons
            WHERE role = 'internal' AND face_signature IS NOT NULL
            ORDER BY display_name COLLATE NOCASE, person_id
            """
        ).fetchall()
        return [
            {
                "person_id": row["person_id"],
                "display_name": row["display_name"],
                "photo_url": (
                    None
                    if row["reference_photo_path"] is None
                    else f"/{row['reference_photo_path']}"
                ),
                "registered_date": row["registered_date"],
            }
            for row in rows
        ]
    finally:
        if close_conn:
            conn.close()


def create_internal_team_member(handler: DashboardHandler) -> dict:
    """Parse a multipart form and save one internal team member."""
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/form-data"):
        return {"ok": False, "error": "Use multipart/form-data with name and photo."}

    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        return {"ok": False, "error": "Invalid upload length."}
    if content_length <= 0:
        return {"ok": False, "error": "Upload is empty."}
    if content_length > 9 * 1024 * 1024:
        return {"ok": False, "error": "Upload must be 9 MB or smaller."}

    form = parse_multipart_form(content_type, handler.rfile.read(content_length))

    name = form.get("name", ("", None))[0].strip()
    photo_bytes = form.get("photo", (b"", None))[0]
    if not name:
        return {"ok": False, "error": "Name is required."}
    if not isinstance(photo_bytes, bytes):
        return {"ok": False, "error": "Photo is required."}

    raw = photo_bytes
    if not raw:
        return {"ok": False, "error": "Photo is empty."}
    if len(raw) > 8 * 1024 * 1024:
        return {"ok": False, "error": "Photo must be 8 MB or smaller."}

    try:
        image = Image.open(io_bytes(raw)).convert("RGB")
    except Exception:
        return {"ok": False, "error": "Photo must be a readable image file."}

    rgb = np.array(image)
    signature, existing_identity = compute_face_signature_and_match(rgb)
    if signature is None:
        return {"ok": False, "error": "No usable face was found in that photo."}

    INTERNAL_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(name)
    person_id = (
        existing_identity.person_id
        if existing_identity is not None
        else f"INTERNAL-{slug.upper()}"
    )
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    photo_path = INTERNAL_PHOTOS_DIR / f"{slug}_{timestamp}.jpg"

    image.thumbnail((800, 800))
    image.save(photo_path, format="JPEG", quality=90)
    relative_photo_path = photo_path.relative_to(PROJECT_DIR).as_posix()

    from database import DatabaseManager

    db = DatabaseManager(DATABASE_PATH)
    ok = db.upsert_internal_person(
        person_id=person_id,
        display_name=name,
        face_signature=signature,
        reference_photo_path=relative_photo_path,
    )
    if not ok:
        return {"ok": False, "error": "Could not save internal team member."}

    return {
        "ok": True,
        "member": {
            "person_id": person_id,
            "display_name": name,
            "photo_url": f"/{relative_photo_path}",
        },
    }


def compute_face_signature_and_match(rgb: np.ndarray):
    """Compute the same embedding signature used by live recognition."""
    from database import DatabaseManager
    from face_identity import FaceIdentityManager

    identity = FaceIdentityManager(DatabaseManager(DATABASE_PATH))
    return identity.compute_signature_and_match(rgb)


def parse_multipart_form(content_type: str, body: bytes) -> dict[str, tuple]:
    """Parse multipart/form-data into {field_name: (value, filename)}."""
    message = BytesParser(policy=default).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + body
    )
    fields = {}
    if not message.is_multipart():
        return fields

    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue

        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            fields[name] = (payload, filename)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = (payload.decode(charset, errors="replace"), None)

    return fields


def slugify(value: str) -> str:
    """Create a stable internal person ID suffix from a display name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "member"


def io_bytes(raw: bytes):
    from io import BytesIO

    return BytesIO(raw)


def main():
    if not DATABASE_PATH.exists():
        raise SystemExit(f"Database not found: {DATABASE_PATH}")

    server = create_server()
    host, port = server.server_address
    print(f"Dashboard running at http://{host}:{port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()


def create_server() -> ThreadingHTTPServer:
    """Bind to PORT, or the next available local port."""
    last_error = None
    for port in range(PORT, PORT + PORT_SCAN_LIMIT):
        try:
            return ThreadingHTTPServer((HOST, port), DashboardHandler)
        except OSError as exc:
            last_error = exc
            if exc.errno != 48:
                raise

    raise OSError(f"No available dashboard port found: {last_error}")


if __name__ == "__main__":
    main()
