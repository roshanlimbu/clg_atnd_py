-- ============================================================
-- Face Attendance System — Database Schema
-- Compatible: SQLite (edge device) / MySQL (cloud server)
-- ============================================================

-- ── Reference: user types ──────────────────────────────────
CREATE TABLE IF NOT EXISTS user_types (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name   TEXT    NOT NULL UNIQUE,   -- 'internal' | 'external'
    description TEXT
);

INSERT OR IGNORE INTO user_types (type_name, description) VALUES
    ('internal', 'Team members — presence tracking only, never counted'),
    ('external', 'Attendees — attendance counting, once per day');

-- ── Reference: angle types ─────────────────────────────────
CREATE TABLE IF NOT EXISTS angle_types (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    angle_name  TEXT NOT NULL UNIQUE,      -- 'front' | 'left_15' | 'right_15' | 'left_30' | 'right_30'
    yaw_degrees INTEGER NOT NULL,
    description TEXT
);

INSERT OR IGNORE INTO angle_types (angle_name, yaw_degrees, description) VALUES
    ('front',     0,  'Frontal face'),
    ('left_15',  -15, 'Slight left profile'),
    ('right_15',  15, 'Slight right profile'),
    ('left_30',  -30, 'Moderate left profile'),
    ('right_30',  30, 'Moderate right profile');

-- ── Reference: attendance statuses ─────────────────────────
CREATE TABLE IF NOT EXISTS attendance_statuses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    status_name TEXT NOT NULL UNIQUE       -- 'counted' | 'debounced' | 'low_confidence' | 'spoofed'
);

INSERT OR IGNORE INTO attendance_statuses (status_name) VALUES
    ('counted'),
    ('debounced'),
    ('low_confidence'),
    ('spoofed');

-- ── Reference: cameras ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name TEXT    NOT NULL,
    location    TEXT,
    rtsp_url    TEXT,
    device_index INTEGER DEFAULT 0,
    is_active   INTEGER DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (DATETIME('now'))
);

INSERT OR IGNORE INTO cameras (camera_name, location, device_index) VALUES
    ('default', 'main-entrance', 0);

-- ── Users (both internal + external) ───────────────────────
CREATE TABLE IF NOT EXISTS users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id           TEXT    NOT NULL UNIQUE,     -- 'person_1', 'person_2', ...
    user_type_id        INTEGER NOT NULL DEFAULT 2,  -- FK → user_types
    display_name        TEXT,
    email               TEXT,
    phone               TEXT,
    registered_date     DATE    NOT NULL DEFAULT (DATE('now')),
    reference_photo_path TEXT,                       -- original enrollment photo
    is_active           INTEGER NOT NULL DEFAULT 1,
    notes               TEXT,
    created_at          TEXT    NOT NULL DEFAULT (DATETIME('now')),
    updated_at          TEXT    NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_type_id) REFERENCES user_types(id)
);

CREATE INDEX IF NOT EXISTS idx_users_person_id ON users(person_id);
CREATE INDEX IF NOT EXISTS idx_users_user_type ON users(user_type_id);
CREATE INDEX IF NOT EXISTS idx_users_registered_date ON users(registered_date);

-- ── Face images (enrollment originals + synthetic angles) ──
CREATE TABLE IF NOT EXISTS face_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    angle_type_id   INTEGER NOT NULL DEFAULT 1,     -- FK → angle_types
    image_path      TEXT    NOT NULL,                -- local storage path
    image_size_bytes INTEGER,
    is_synthetic    INTEGER NOT NULL DEFAULT 0,      -- 0=original, 1=3DDFA_V2 generated
    created_at      TEXT    NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_id)       REFERENCES users(id),
    FOREIGN KEY (angle_type_id) REFERENCES angle_types(id)
);

CREATE INDEX IF NOT EXISTS idx_face_images_user ON face_images(user_id);
CREATE INDEX IF NOT EXISTS idx_face_images_angle ON face_images(angle_type_id);

-- ── Face embeddings (512-d ArcFace vectors) ────────────────
CREATE TABLE IF NOT EXISTS face_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    angle_type_id   INTEGER NOT NULL DEFAULT 1,     -- which angle this embedding came from
    embedding_json  TEXT    NOT NULL,                -- JSON array of 512 floats
    faiss_position  INTEGER,                        -- position in FAISS index (-1 if not indexed)
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (user_id)       REFERENCES users(id),
    FOREIGN KEY (angle_type_id) REFERENCES angle_types(id)
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_user ON face_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_faiss ON face_embeddings(faiss_position);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_active ON face_embeddings(is_active);

-- ── FAISS index metadata ──────────────────────────────────
CREATE TABLE IF NOT EXISTS faiss_index_meta (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    index_path      TEXT    NOT NULL,                -- path to .index file
    id_map_path     TEXT    NOT NULL,                -- path to id_map.json
    total_vectors   INTEGER NOT NULL DEFAULT 0,
    dimension       INTEGER NOT NULL DEFAULT 512,
    last_rebuilt_at TEXT    NOT NULL DEFAULT (DATETIME('now')),
    is_current      INTEGER NOT NULL DEFAULT 1
);

-- ── Detection log (every face, every frame) ────────────────
CREATE TABLE IF NOT EXISTS detection_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,                        -- NULL for unknown faces
    camera_id       INTEGER NOT NULL DEFAULT 1,
    status_id       INTEGER NOT NULL,               -- FK → attendance_statuses
    confidence      REAL    NOT NULL DEFAULT 0.0,
    face_crop_path  TEXT,                           -- saved face crop image
    bbox_x1         INTEGER,
    bbox_y1         INTEGER,
    bbox_x2         INTEGER,
    bbox_y2         INTEGER,
    is_spoofed      INTEGER NOT NULL DEFAULT 0,
    detected_at     TEXT    NOT NULL DEFAULT (DATETIME('now')),
    synced          INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id)   REFERENCES users(id),
    FOREIGN KEY (camera_id) REFERENCES cameras(id),
    FOREIGN KEY (status_id) REFERENCES attendance_statuses(id)
);

CREATE INDEX IF NOT EXISTS idx_detection_logs_user    ON detection_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_logs_time    ON detection_logs(detected_at);
CREATE INDEX IF NOT EXISTS idx_detection_logs_camera  ON detection_logs(camera_id);
CREATE INDEX IF NOT EXISTS idx_detection_logs_synced  ON detection_logs(synced);
CREATE INDEX IF NOT EXISTS idx_detection_logs_status  ON detection_logs(status_id);

-- ── Internal presence log (every crossing event) ──────────
CREATE TABLE IF NOT EXISTS internal_presence_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    camera_id       INTEGER NOT NULL DEFAULT 1,
    confidence      REAL    NOT NULL DEFAULT 0.0,
    face_crop_path  TEXT,
    detected_at     TEXT    NOT NULL DEFAULT (DATETIME('now')),
    synced          INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id)   REFERENCES users(id),
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

CREATE INDEX IF NOT EXISTS idx_internal_presence_user ON internal_presence_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_internal_presence_time ON internal_presence_logs(detected_at);
CREATE INDEX IF NOT EXISTS idx_internal_presence_synced ON internal_presence_logs(synced);

-- ── Internal presence summary (daily crossing counter) ────
CREATE TABLE IF NOT EXISTS internal_presence_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    date            DATE    NOT NULL,
    crossing_count  INTEGER NOT NULL DEFAULT 1,
    first_seen      TIME    NOT NULL,
    last_seen       TIME    NOT NULL,
    UNIQUE(user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_internal_presence_summary_user ON internal_presence_summary(user_id);
CREATE INDEX IF NOT EXISTS idx_internal_presence_summary_date ON internal_presence_summary(date);

-- ── External attendance log (one row per person per day) ──
CREATE TABLE IF NOT EXISTS attendance_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    date            DATE    NOT NULL,
    first_seen      TIME    NOT NULL,
    last_seen       TIME    NOT NULL,
    count           INTEGER NOT NULL DEFAULT 1,      -- times seen today
    confidence      REAL,
    face_crop_path  TEXT,                           -- best photo from today
    synced          INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_attendance_logs_user  ON attendance_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_attendance_logs_date  ON attendance_logs(date);
CREATE INDEX IF NOT EXISTS idx_attendance_logs_synced ON attendance_logs(synced);

-- ── Security flags (spoofed / suspicious) ─────────────────
CREATE TABLE IF NOT EXISTS security_flags (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id    INTEGER,
    flag_type       TEXT    NOT NULL,                -- 'spoof' | 'tamper' | 'low_confidence_repeat'
    severity        TEXT    NOT NULL DEFAULT 'low',  -- 'low' | 'medium' | 'high'
    details         TEXT,
    face_crop_path  TEXT,
    flagged_at      TEXT    NOT NULL DEFAULT (DATETIME('now')),
    reviewed        INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (detection_id) REFERENCES detection_logs(id)
);

CREATE INDEX IF NOT EXISTS idx_security_flags_type   ON security_flags(flag_type);
CREATE INDEX IF NOT EXISTS idx_security_flags_review ON security_flags(reviewed);

-- ── Cooldown tracker (SQLite-backed for crash survival) ──
CREATE TABLE IF NOT EXISTS cooldown_tracker (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    camera_id       INTEGER NOT NULL DEFAULT 1,
    last_seen_at    TEXT    NOT NULL,
    UNIQUE(user_id, camera_id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ── Verification ──────────────────────────────────────────
SELECT 'Schema created successfully.' AS status;
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;
