"""
Local dashboard server for the face attendance system.

Run from the project root with:
    ../venv312/bin/python frontend/server.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FRONTEND_DIR.parent
DATABASE_PATH = PROJECT_DIR / "attendance.db"
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

        if parsed.path.startswith("/attendance_photos/"):
            self._send_attendance_photo(parsed.path)
            return

        super().do_GET()

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

    def _send_attendance_photo(self, request_path: str):
        relative_path = unquote(request_path).lstrip("/")
        photo_root = (PROJECT_DIR / "attendance_photos").resolve()
        photo_path = (PROJECT_DIR / relative_path).resolve()

        if photo_path != photo_root and photo_root not in photo_path.parents:
            self.send_error(404)
            return

        if not photo_path.is_file():
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(photo_path)))
        self.send_header("Content-Length", str(photo_path.stat().st_size))
        self.end_headers()
        with photo_path.open("rb") as file_obj:
            self.copyfile(file_obj, self.wfile)


def get_connection() -> sqlite3.Connection:
    """Open a short-lived read-only SQLite connection."""
    uri = f"file:{DATABASE_PATH}?mode=ro&cache=shared"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def get_dashboard_summary() -> dict:
    """Read the latest attendance state for the dashboard."""
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
                    {photo_select}
                FROM attendance AS a
                JOIN persons AS p ON p.person_id = a.person_id
                WHERE a.date = ? AND p.face_signature IS NOT NULL
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
        }


def serialize_face(row: sqlite3.Row) -> dict:
    """Convert a SQLite row to JSON-safe dashboard data."""
    confidence = row["confidence"]
    photo_path = row["photo_path"]
    return {
        "person_id": row["person_id"],
        "date": row["date"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "count": int(row["count"]),
        "confidence": None if confidence is None else round(float(confidence), 4),
        "photo_url": None if photo_path is None else f"/{photo_path}",
    }


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
