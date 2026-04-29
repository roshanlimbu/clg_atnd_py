"""
Generated face identity matching.

This replaces fixed-label model recognition for event/festival attendance:
each new face crop gets a generated ID, and later similar crops reuse that ID.

Uses dlib's ResNet-34 face embedding model (via face_recognition) which
produces a stable 128-dimensional vector per face. Two crops of the same
person will have L2 distance < 0.5; two different people will be > 0.6.
This is far more accurate than the previous DCT perceptual hash approach.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class FaceIdentityResult:
    """Generated identity result for one detected face."""

    person_id: str
    confidence: float
    is_new: bool = False
    unknown: bool = False
    above_threshold: bool = True
    display_name: Optional[str] = None
    role: str = "guest"

    @property
    def is_unknown(self) -> bool:
        return self.unknown

    @property
    def is_above_threshold(self) -> bool:
        return self.above_threshold

    @property
    def is_internal(self) -> bool:
        return self.role == "internal"


@dataclass
class KnownFace:
    """Stored identity embedding plus metadata."""

    person_id: str
    embedding: np.ndarray
    display_name: Optional[str] = None
    role: str = "guest"


class FaceIdentityManager:
    """
    Assign generated IDs to new faces using dlib ResNet-34 face embeddings.

    Each face crop is encoded into a 128-dimensional vector. New faces whose
    nearest known vector is farther than MATCH_DISTANCE_THRESHOLD are
    registered as new persons; otherwise the existing ID is reused.

    Lower MATCH_DISTANCE_THRESHOLD = stricter (fewer false matches).
    Typical operating range: 0.45-0.60. Default 0.58 reduces duplicate IDs
    for live camera crops while staying below face_recognition's 0.60 default.
    """

    MATCH_DISTANCE_THRESHOLD = 0.58
    CACHE_REFRESH_SECONDS = 10

    def __init__(self, db_manager: DatabaseManager):
        self._face_recognition = self._import_face_recognition()
        self.db = db_manager
        self._known: List[KnownFace] = self._load_known_embeddings()
        self._last_refresh_at = time.monotonic()
        logger.info("Face identity cache loaded: %s known faces", len(self._known))

    # ── Public API ────────────────────────────────────────────────────────

    def identify_batch(
        self, face_images: Sequence[np.ndarray]
    ) -> List[FaceIdentityResult]:
        """Identify or register a batch of aligned RGB face images."""
        self._refresh_known_embeddings_if_needed()
        return [self.identify(image) for image in face_images]

    def identify(self, face_image: np.ndarray) -> FaceIdentityResult:
        """Identify an aligned RGB face image, creating a generated ID if needed."""
        embedding = self._compute_embedding(face_image)
        if embedding is None:
            # Do not register detector false positives or unusable crops as
            # new people. They should be ignored by the attendance layer.
            return FaceIdentityResult(
                person_id="Unknown",
                confidence=0.0,
                unknown=True,
                above_threshold=False,
            )

        match = self._find_best_match(embedding)
        if match is not None:
            known_face, confidence = match
            # Incrementally update the stored embedding toward the new one so
            # it adapts to natural changes in lighting/angle over time.
            if known_face.role != "internal":
                self._update_embedding(known_face.person_id, embedding)
            return FaceIdentityResult(
                person_id=known_face.person_id,
                confidence=confidence,
                is_new=False,
                display_name=known_face.display_name,
                role=known_face.role,
            )

        person_id = self._create_person(embedding)
        return FaceIdentityResult(person_id=person_id, confidence=1.0, is_new=True)

    def compute_signature(self, face_image: np.ndarray) -> Optional[str]:
        """Compute a serialized face embedding for uploaded reference images."""
        embedding = self._compute_embedding(face_image)
        if embedding is None:
            return None
        return self._serialize(embedding)

    def compute_signature_and_match(
        self, face_image: np.ndarray
    ) -> tuple[Optional[str], Optional[FaceIdentityResult]]:
        """Compute a reference signature and find an existing matching identity."""
        embedding = self._compute_embedding(face_image)
        if embedding is None:
            return None, None

        match = self._find_best_match(embedding)
        if match is None:
            return self._serialize(embedding), None

        known_face, confidence = match
        return self._serialize(embedding), FaceIdentityResult(
            person_id=known_face.person_id,
            confidence=confidence,
            display_name=known_face.display_name,
            role=known_face.role,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _import_face_recognition():
        try:
            import face_recognition as fr
            return fr
        except ImportError:
            raise ImportError(
                "face-recognition is not installed. "
                "Run: pip install face-recognition  (requires cmake: brew install cmake)"
            )

    def _load_known_embeddings(self) -> List[KnownFace]:
        result = []
        for person in self.db.get_all_persons():
            sig = person["face_signature"]
            emb = self._deserialize(sig)
            if emb is not None:
                result.append(KnownFace(
                    person_id=person["person_id"],
                    embedding=emb,
                    display_name=person["display_name"],
                    role=person["role"] or "guest",
                ))
        return result

    def _refresh_known_embeddings_if_needed(self):
        """Pick up identities added from the dashboard while the camera runs."""
        now = time.monotonic()
        if now - self._last_refresh_at < self.CACHE_REFRESH_SECONDS:
            return

        self._known = self._load_known_embeddings()
        self._last_refresh_at = now
        logger.debug("Face identity cache refreshed: %s known faces", len(self._known))

    def _create_person(self, embedding: Optional[np.ndarray]) -> str:
        serialized = self._serialize(embedding) if embedding is not None else None
        for _ in range(5):
            person_id = self.db.get_next_face_id(date.today())
            if self.db.add_person(person_id, serialized):
                if embedding is not None:
                    self._known.append(KnownFace(
                        person_id=person_id,
                        embedding=embedding,
                    ))
                logger.info("Registered new face ID: %s", person_id)
                return person_id
        raise RuntimeError("Could not allocate a unique face ID")

    def _update_embedding(self, person_id: str, new_embedding: np.ndarray) -> None:
        """Exponential moving average update: blends new embedding into stored one."""
        alpha = 0.1  # weight given to the new sample; 0.1 = slow, stable adaptation
        for known_face in self._known:
            if known_face.person_id == person_id:
                updated = (1.0 - alpha) * known_face.embedding + alpha * new_embedding
                known_face.embedding = updated
                self.db.update_person_signature(person_id, self._serialize(updated))
                break

    def _find_best_match(
        self, embedding: np.ndarray
    ) -> Optional[Tuple[KnownFace, float]]:
        if not self._known:
            return None

        known_encs = [known_face.embedding for known_face in self._known]

        distances = self._face_recognition.face_distance(known_encs, embedding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance > self.MATCH_DISTANCE_THRESHOLD:
            return None

        confidence = float(1.0 - best_distance)
        return self._known[best_idx], confidence

    def _compute_embedding(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode a pre-cropped RGB face image into a 128-d dlib embedding.

        The upstream detector may return a loose person/object crop, so first
        locate the actual face inside the crop. Forcing the whole crop to be a
        face makes embeddings unstable and creates duplicate generated IDs.
        """
        if face_rgb is None or face_rgb.size == 0:
            raise ValueError("Cannot encode an empty face image")

        if face_rgb.ndim != 3 or face_rgb.shape[2] != 3:
            logger.debug("Invalid face crop shape for encoding: %s", face_rgb.shape)
            return None

        h, w = face_rgb.shape[:2]
        if h < 20 or w < 20:
            face_rgb = cv2.resize(face_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
            h, w = face_rgb.shape[:2]

        locations = self._face_recognition.face_locations(
            face_rgb,
            number_of_times_to_upsample=1,
            model="hog",
        )
        if not locations:
            logger.debug("No real face found inside detector crop")
            return None

        locations = [self._largest_location(locations)]
        encodings = self._face_recognition.face_encodings(
            face_rgb,
            known_face_locations=locations,
            num_jitters=3,   # average 3 jittered crops → more stable embedding
            model="large",   # 68-landmark model, more accurate than "small"
        )

        if not encodings:
            logger.debug("face_recognition returned no encoding for this crop")
            return None

        return encodings[0]  # shape (128,)

    @staticmethod
    def _largest_location(
        locations: Sequence[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """Return the largest dlib face location: (top, right, bottom, left)."""
        return max(
            locations,
            key=lambda loc: max(0, loc[2] - loc[0]) * max(0, loc[1] - loc[3]),
        )

    @staticmethod
    def _serialize(embedding: np.ndarray) -> str:
        """Store embedding as a compact JSON float array."""
        return json.dumps(embedding.tolist(), separators=(",", ":"))

    @staticmethod
    def _deserialize(signature: Optional[str]) -> Optional[np.ndarray]:
        """Load a stored JSON embedding, returning None for old hex-hash signatures."""
        if not signature:
            return None
        try:
            data = json.loads(signature)
            if isinstance(data, list) and len(data) == 128:
                return np.array(data, dtype=np.float64)
        except (json.JSONDecodeError, ValueError):
            pass
        # Old DCT hex signatures are silently ignored; those persons will get
        # a new embedding registered on their next sighting.
        return None
