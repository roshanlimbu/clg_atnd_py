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

    @property
    def is_unknown(self) -> bool:
        return self.unknown

    @property
    def is_above_threshold(self) -> bool:
        return self.above_threshold


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

    def __init__(self, db_manager: DatabaseManager):
        self._face_recognition = self._import_face_recognition()
        self.db = db_manager
        self._known: List[Tuple[str, np.ndarray]] = self._load_known_embeddings()
        logger.info("Face identity cache loaded: %s known faces", len(self._known))

    # ── Public API ────────────────────────────────────────────────────────

    def identify_batch(
        self, face_images: Sequence[np.ndarray]
    ) -> List[FaceIdentityResult]:
        """Identify or register a batch of aligned RGB face images."""
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
            person_id, confidence = match
            # Incrementally update the stored embedding toward the new one so
            # it adapts to natural changes in lighting/angle over time.
            self._update_embedding(person_id, embedding)
            return FaceIdentityResult(person_id=person_id, confidence=confidence, is_new=False)

        person_id = self._create_person(embedding)
        return FaceIdentityResult(person_id=person_id, confidence=1.0, is_new=True)

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

    def _load_known_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        result = []
        for person in self.db.get_all_persons():
            sig = person["face_signature"]
            emb = self._deserialize(sig)
            if emb is not None:
                result.append((person["person_id"], emb))
        return result

    def _create_person(self, embedding: Optional[np.ndarray]) -> str:
        serialized = self._serialize(embedding) if embedding is not None else None
        for _ in range(5):
            person_id = self.db.get_next_face_id(date.today())
            if self.db.add_person(person_id, serialized):
                if embedding is not None:
                    self._known.append((person_id, embedding))
                logger.info("Registered new face ID: %s", person_id)
                return person_id
        raise RuntimeError("Could not allocate a unique face ID")

    def _update_embedding(self, person_id: str, new_embedding: np.ndarray) -> None:
        """Exponential moving average update: blends new embedding into stored one."""
        alpha = 0.1  # weight given to the new sample; 0.1 = slow, stable adaptation
        for i, (pid, stored_emb) in enumerate(self._known):
            if pid == person_id:
                updated = (1.0 - alpha) * stored_emb + alpha * new_embedding
                self._known[i] = (person_id, updated)
                self.db.update_person_signature(person_id, self._serialize(updated))
                break

    def _find_best_match(
        self, embedding: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        if not self._known:
            return None

        known_encs = [enc for _, enc in self._known]
        person_ids = [pid for pid, _ in self._known]

        distances = self._face_recognition.face_distance(known_encs, embedding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance > self.MATCH_DISTANCE_THRESHOLD:
            return None

        confidence = float(1.0 - best_distance)
        return person_ids[best_idx], confidence

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
