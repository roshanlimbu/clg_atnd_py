"""
========================================================
tracker.py — Lightweight Face Tracker for Walking Crowds
========================================================

Tracks faces across frames using centroid matching.
Solves three problems at once:

1. Recognize each person ONCE per crossing (not every frame)
2. Pick the BEST quality frame per person for ArcFace
3. Skip faces too small/blurry for reliable recognition

Instead of running ArcFace on every detected face every frame,
the tracker waits for a sharp front-facing crop and sends only
that to ArcFace. Cached results are reused for display.
========================================================
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    """Single tracked face across multiple frames."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]

    # Best quality face crop seen so far (for ArcFace)
    best_crop: Optional[np.ndarray] = None
    best_quality: float = 0.0

    # Current frame's aligned crop (for display/fallback)
    current_crop: Optional[np.ndarray] = None

    # Recognition result (cached after first ArcFace call)
    recognition_result: object = None  # FaceIdentityResult
    is_recognized: bool = False
    recognized_at_frame: int = 0

    # Lifecycle counters
    frames_seen: int = 0
    frames_since_last_seen: int = 0
    created_at_frame: int = 0
    detection_confidence: float = 0.0


class FaceTracker:
    """
    Centroid-based face tracker with quality-gated recognition.

    Workflow per frame:
        1. update() — match new detections to existing tracks
        2. get_tracks_to_recognize() — which tracks need ArcFace?
        3. store_recognition() — cache ArcFace results
        4. get_active_tracks() — all tracks for display + attendance

    Recognition is only triggered when:
        - Track has been seen for MIN_FRAMES_BEFORE_RECOGNIZE frames
        - Best crop quality exceeds MIN_QUALITY threshold
        - Track hasn't been recognized yet (or needs re-recognition)
    """

    # ── Tunable parameters ────────────────────────────────────────────
    # Max frames a track survives without being seen
    MAX_DISAPPEARED = 12

    # Max centroid distance (pixels) to match detection to track
    MATCH_DISTANCE = 100

    # Min frames before triggering recognition (wait for good crop)
    MIN_FRAMES_BEFORE_RECOGNIZE = 2

    # Min Laplacian sharpness to consider a crop worth recognizing
    MIN_QUALITY = 20.0

    # Min face size (pixels) for reliable ArcFace recognition
    MIN_FACE_SIZE_FOR_RECOGNITION = 60

    # Re-recognize after this many frames (for long-staying faces)
    RE_RECOGNIZE_INTERVAL = 90

    def __init__(self):
        self._tracks: Dict[int, FaceTrack] = {}
        self._next_id: int = 0
        self._frame_count: int = 0

    # ── Main API ──────────────────────────────────────────────────────

    def update(self, detected_faces) -> None:
        """
        Match new detections to existing tracks.

        Args:
            detected_faces: List of DetectedFace objects from detector.
        """
        self._frame_count += 1

        if not detected_faces:
            # No detections — increment disappeared for all tracks
            for track in list(self._tracks.values()):
                track.frames_since_last_seen += 1
                if track.frames_since_last_seen > self.MAX_DISAPPEARED:
                    self._close_track(track.track_id)
            return

        # Compute centroids of new detections
        det_centroids = []
        for face in detected_faces:
            cx = (face.bbox[0] + face.bbox[2]) // 2
            cy = (face.bbox[1] + face.bbox[3]) // 2
            det_centroids.append((cx, cy))

        if not self._tracks:
            # No existing tracks — create one per detection
            for i, face in enumerate(detected_faces):
                self._create_track(face, det_centroids[i])
            return

        # Match detections to existing tracks (greedy nearest-centroid)
        track_ids = list(self._tracks.keys())
        track_centroids = [
            self._tracks[tid].centroid for tid in track_ids
        ]

        matched_tracks = set()
        matched_dets = set()

        # Build distance matrix and match greedily
        pairs = []
        for di, dc in enumerate(det_centroids):
            for ti, tc in enumerate(track_centroids):
                dist = ((dc[0] - tc[0]) ** 2 + (dc[1] - tc[1]) ** 2) ** 0.5
                if dist < self.MATCH_DISTANCE:
                    pairs.append((dist, di, ti))

        pairs.sort(key=lambda x: x[0])

        for dist, di, ti in pairs:
            if di in matched_dets or ti in matched_tracks:
                continue
            tid = track_ids[ti]
            self._update_track(tid, detected_faces[di], det_centroids[di])
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Create new tracks for unmatched detections
        for di in range(len(detected_faces)):
            if di not in matched_dets:
                self._create_track(detected_faces[di], det_centroids[di])

        # Increment disappeared for unmatched tracks
        for ti in range(len(track_ids)):
            if ti not in matched_tracks:
                tid = track_ids[ti]
                self._tracks[tid].frames_since_last_seen += 1
                if self._tracks[tid].frames_since_last_seen > self.MAX_DISAPPEARED:
                    self._close_track(tid)

    def get_tracks_to_recognize(self) -> List[FaceTrack]:
        """
        Return tracks that should be sent to ArcFace.

        A track qualifies when:
        1. It has been seen for enough frames (not a flash)
        2. Its best crop quality exceeds the minimum threshold
        3. Its face is large enough for reliable recognition
        4. It hasn't been recognized yet (or is due for re-recognition)
        """
        result = []
        for track in self._tracks.values():
            if track.frames_since_last_seen > 0:
                continue  # Only consider currently visible tracks

            # Already recognized and not due for refresh
            if track.is_recognized:
                frames_since = self._frame_count - track.recognized_at_frame
                if frames_since < self.RE_RECOGNIZE_INTERVAL:
                    continue

            # Wait for enough frames to pick a good crop
            if track.frames_seen < self.MIN_FRAMES_BEFORE_RECOGNIZE:
                continue

            # Quality gate
            if track.best_quality < self.MIN_QUALITY:
                continue

            # Size gate
            w = track.bbox[2] - track.bbox[0]
            h = track.bbox[3] - track.bbox[1]
            if w < self.MIN_FACE_SIZE_FOR_RECOGNITION or h < self.MIN_FACE_SIZE_FOR_RECOGNITION:
                continue

            if track.best_crop is not None:
                result.append(track)

        return result

    def store_recognition(self, track_id: int, result) -> None:
        """Cache an ArcFace recognition result for a track."""
        if track_id in self._tracks:
            track = self._tracks[track_id]
            track.recognition_result = result
            track.is_recognized = True
            track.recognized_at_frame = self._frame_count

    def get_active_tracks(self) -> List[FaceTrack]:
        """Return all currently visible tracks (for display + attendance)."""
        return [
            t for t in self._tracks.values()
            if t.frames_since_last_seen == 0
        ]

    def get_newly_recognized_track_ids(self) -> List[int]:
        """Return track IDs that were just recognized this frame."""
        return [
            t.track_id for t in self._tracks.values()
            if t.is_recognized and t.recognized_at_frame == self._frame_count
        ]

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.frames_since_last_seen == 0)

    @property
    def total_tracks(self) -> int:
        return len(self._tracks)

    # ── Internal methods ──────────────────────────────────────────────

    def _create_track(self, face, centroid) -> int:
        tid = self._next_id
        self._next_id += 1

        quality = self._compute_quality(face.aligned_image, face.bbox)

        self._tracks[tid] = FaceTrack(
            track_id=tid,
            bbox=face.bbox,
            centroid=centroid,
            best_crop=face.aligned_image,
            best_quality=quality,
            current_crop=face.aligned_image,
            frames_seen=1,
            frames_since_last_seen=0,
            created_at_frame=self._frame_count,
            detection_confidence=face.detection_confidence,
        )
        return tid

    def _update_track(self, track_id: int, face, centroid) -> None:
        track = self._tracks[track_id]
        track.bbox = face.bbox
        track.centroid = centroid
        track.current_crop = face.aligned_image
        track.frames_seen += 1
        track.frames_since_last_seen = 0
        track.detection_confidence = face.detection_confidence

        # Update best crop if this frame is better quality
        quality = self._compute_quality(face.aligned_image, face.bbox)
        if quality > track.best_quality:
            track.best_crop = face.aligned_image
            track.best_quality = quality

    def _close_track(self, track_id: int) -> None:
        if track_id in self._tracks:
            track = self._tracks[track_id]
            logger.debug(
                "Track %d closed: seen %d frames, recognized=%s",
                track_id, track.frames_seen, track.is_recognized,
            )
            del self._tracks[track_id]

    @staticmethod
    def _compute_quality(face_crop: np.ndarray, bbox: tuple) -> float:
        """
        Score face quality: sharpness weighted by face size.

        Higher score = sharper image of a larger face = better for ArcFace.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        try:
            if face_crop.ndim == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_crop

            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Size factor: larger faces are more reliable
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            size_factor = min(1.0, (w * h) / (120.0 * 120.0))

            return sharpness * (0.5 + 0.5 * size_factor)
        except Exception:
            return 0.0
