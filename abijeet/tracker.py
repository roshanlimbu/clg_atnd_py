"""
========================================================
tracker.py — Face Tracker with Three-Zone Routing
========================================================

Tracks faces across frames using centroid matching.
Now includes three-zone confidence routing:

Zone 1 — MATCHED (confidence >= 0.55)
  Internal team → silent pass (nothing logged)
  External known → normal attendance flow

Zone 2 — UNCERTAIN (confidence 0.30–0.55)
  Hold in buffer, wait for better frame
  If still uncertain after MAX_UNCERTAIN_FRAMES → discard
  If confidence rises above 0.55 → reclassify as matched

Zone 3 — NEW FACE (confidence < 0.30 for 3+ frames)
  Genuinely new person not in the system
  After MIN_CONFIRM_FRAMES → auto-register as visitor
  FAISS updated immediately to prevent duplicates

Solves:
1. Recognize each person ONCE per crossing (not every frame)
2. Pick the BEST quality frame per person for ArcFace
3. Skip faces too small/blurry for reliable recognition
4. Never accidentally register internal team at bad angles
5. Auto-register genuinely new visitors after confirmation
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

    # ── Three-zone routing state ──────────────────────────────
    zone: str = (
        "pending"  # "pending" | "matched" | "uncertain" | "new_face" | "silent_pass" | "registered"
    )
    zone_frames: int = 0  # How many frames in current zone
    best_raw_similarity: float = 0.0  # Best FAISS similarity seen
    worst_raw_similarity: float = (
        1.0  # Worst FAISS similarity (for new_face confirmation)
    )
    is_auto_registered: bool = False


class FaceTracker:
    """
    Centroid-based face tracker with three-zone confidence routing.

    Workflow per frame:
        1. update() — match new detections to existing tracks
        2. get_tracks_to_recognize() — which tracks need ArcFace?
        3. store_recognition() — cache ArcFace results + zone routing
        4. get_tracks_to_register() — NEW: which tracks are confirmed new faces?
        5. mark_registered() — NEW: after registration, update track state
        6. get_active_tracks() — all tracks for display + attendance
    """

    # ── Tunable parameters ────────────────────────────────────────────
    MAX_DISAPPEARED = 12
    MATCH_DISTANCE = 100
    MIN_FRAMES_BEFORE_RECOGNIZE = 2
    MIN_QUALITY = 20.0
    MIN_FACE_SIZE_FOR_RECOGNITION = 60
    RE_RECOGNIZE_INTERVAL = 90

    # ── Zone-specific parameters ──────────────────────────────────────
    # Minimum frames a new face must be confirmed before auto-registration
    MIN_CONFIRM_FRAMES = 3
    # Maximum frames to hold an uncertain face before discarding
    MAX_UNCERTAIN_FRAMES = 10

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
            for i, face in enumerate(detected_faces):
                self._create_track(face, det_centroids[i])
            return

        # Match detections to existing tracks (greedy nearest-centroid)
        track_ids = list(self._tracks.keys())
        track_centroids = [self._tracks[tid].centroid for tid in track_ids]

        matched_tracks = set()
        matched_dets = set()

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
        5. It hasn't been silently passed or auto-registered
        """
        result = []
        for track in self._tracks.values():
            if track.frames_since_last_seen > 0:
                continue

            # Skip tracks that are already resolved
            if track.zone in ("silent_pass", "registered"):
                continue

            # Already recognized and not due for refresh
            if track.is_recognized:
                frames_since = self._frame_count - track.recognized_at_frame
                if frames_since < self.RE_RECOGNIZE_INTERVAL:
                    # Re-recognize uncertain AND new_face tracks every frame:
                    # - uncertain: waiting for a better angle to confirm identity
                    # - new_face:  zone_frames must increment to reach MIN_CONFIRM_FRAMES=3
                    #              so the tracker can trigger auto-registration
                    if track.zone not in ("uncertain", "new_face"):
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
            if (
                w < self.MIN_FACE_SIZE_FOR_RECOGNITION
                or h < self.MIN_FACE_SIZE_FOR_RECOGNITION
            ):
                continue

            if track.best_crop is not None:
                result.append(track)

        return result

    def store_recognition(self, track_id: int, result) -> None:
        """
        Cache an ArcFace recognition result and apply zone routing.

        Zone routing logic:
          zone="matched" + internal → silent_pass (completely invisible)
          zone="matched" + external → normal attendance flow
          zone="uncertain"         → hold, re-recognize next frame
          zone="new_face"          → increment confirmation counter
        """
        if track_id not in self._tracks:
            return

        track = self._tracks[track_id]
        track.recognition_result = result
        track.is_recognized = True
        track.recognized_at_frame = self._frame_count

        zone = getattr(result, "zone", "unknown")
        raw_sim = getattr(result, "raw_similarity", 0.0)

        # Update similarity tracking
        track.best_raw_similarity = max(track.best_raw_similarity, raw_sim)
        track.worst_raw_similarity = min(track.worst_raw_similarity, raw_sim)

        # ── Zone routing ──────────────────────────────────────────
        if zone == "matched":
            role = getattr(result, "role", "guest")
            if role == "internal":
                # SILENT PASS — internal team member, completely invisible
                track.zone = "silent_pass"
                logger.debug(
                    "Track %d: SILENT PASS — internal team (%s)",
                    track_id,
                    getattr(result, "display_name", None) or result.person_id,
                )
            else:
                # Normal matched external person
                track.zone = "matched"
                track.zone_frames = 0

        elif zone == "uncertain":
            if track.zone != "uncertain":
                # First time entering uncertainty zone
                track.zone = "uncertain"
                track.zone_frames = 1
            else:
                track.zone_frames += 1

            # Check if uncertainty has resolved via better similarity
            if track.best_raw_similarity >= 0.55:
                # A previous or current frame showed confident match
                # but this frame was uncertain — keep as uncertain
                # (the re-recognize logic will try again next frame)
                pass

            # Check if we should discard this uncertain track
            if track.zone_frames >= self.MAX_UNCERTAIN_FRAMES:
                logger.debug(
                    "Track %d: DISCARDED — uncertain for %d frames (best_sim=%.3f)",
                    track_id,
                    track.zone_frames,
                    track.best_raw_similarity,
                )
                track.zone = "discarded"

        elif zone == "new_face":
            if track.zone != "new_face":
                track.zone = "new_face"
                track.zone_frames = 1
            else:
                track.zone_frames += 1

            # Safety: if similarity suddenly rises, this was probably
            # an internal team member seen from a bad initial angle
            if raw_sim >= 0.55:
                track.zone = "matched"
                track.zone_frames = 0
                logger.debug(
                    "Track %d: NEW_FACE → MATCHED (similarity rose to %.3f)",
                    track_id,
                    raw_sim,
                )
            elif raw_sim >= 0.30:
                track.zone = "uncertain"
                track.zone_frames = 1
                logger.debug(
                    "Track %d: NEW_FACE → UNCERTAIN (similarity rose to %.3f)",
                    track_id,
                    raw_sim,
                )

    def get_tracks_to_register(self) -> List[FaceTrack]:
        """
        Return tracks confirmed as genuinely new faces ready for registration.

        A track qualifies when:
        - zone == "new_face"
        - zone_frames >= MIN_CONFIRM_FRAMES (seen in 3+ recognition passes)
        - best_raw_similarity < 0.30 consistently
        - Not already auto-registered
        """
        result = []
        for track in self._tracks.values():
            if track.zone != "new_face":
                continue
            if track.is_auto_registered:
                continue
            if track.zone_frames < self.MIN_CONFIRM_FRAMES:
                # logger.debug("Track %d new_face but only %d frames", track.track_id, track.zone_frames)
                continue
            # Final safety check: ensure best similarity stayed below threshold
            if track.best_raw_similarity >= 0.30:
                logger.debug(
                    "Track %d rejected for pending registration: sim %.3f",
                    track.track_id,
                    track.best_raw_similarity,
                )
                continue
            if track.best_crop is not None:
                logger.info("Track %d ready for pending registration!", track.track_id)
                result.append(track)
        return result

    def mark_registered(self, track_id: int, result) -> None:
        """
        Mark a track as successfully auto-registered.

        Args:
            track_id: The track that was registered
            result: FaceIdentityResult from register_new_face()
        """
        if track_id in self._tracks:
            track = self._tracks[track_id]
            track.is_auto_registered = True
            track.zone = "registered"
            track.recognition_result = result
            logger.info(
                "Track %d: AUTO-REGISTERED as %s",
                track_id,
                result.person_id,
            )

    def get_active_tracks(self) -> List[FaceTrack]:
        """Return all currently visible tracks (for display + attendance)."""
        return [t for t in self._tracks.values() if t.frames_since_last_seen == 0]

    def get_silent_pass_tracks(self) -> List[FaceTrack]:
        """Return tracks that are internal team — completely invisible."""
        return [
            t
            for t in self._tracks.values()
            if t.zone == "silent_pass" and t.frames_since_last_seen == 0
        ]

    def get_displayable_tracks(self) -> List[FaceTrack]:
        """
        Return active tracks EXCLUDING silent_pass (internal team).
        These are the faces that should appear on the display overlay.
        """
        return [
            t
            for t in self._tracks.values()
            if t.frames_since_last_seen == 0 and t.zone != "silent_pass"
        ]

    def get_newly_recognized_track_ids(self) -> List[int]:
        """Return track IDs that were just recognized this frame."""
        return [
            t.track_id
            for t in self._tracks.values()
            if t.is_recognized and t.recognized_at_frame == self._frame_count
        ]

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.frames_since_last_seen == 0)

    @property
    def total_tracks(self) -> int:
        return len(self._tracks)

    @property
    def silent_pass_count(self) -> int:
        """Number of internal team members currently being silently ignored."""
        return sum(
            1
            for t in self._tracks.values()
            if t.zone == "silent_pass" and t.frames_since_last_seen == 0
        )

    @property
    def uncertain_count(self) -> int:
        """Number of faces currently in the uncertainty buffer."""
        return sum(
            1
            for t in self._tracks.values()
            if t.zone == "uncertain" and t.frames_since_last_seen == 0
        )

    @property
    def pending_registration_count(self) -> int:
        """Number of new faces accumulating confirmation frames."""
        return sum(
            1
            for t in self._tracks.values()
            if t.zone == "new_face"
            and not t.is_auto_registered
            and t.frames_since_last_seen == 0
        )

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
                "Track %d closed: seen %d frames, zone=%s, recognized=%s",
                track_id,
                track.frames_seen,
                track.zone,
                track.is_recognized,
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
