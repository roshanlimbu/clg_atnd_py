"""
Face recognition module for the attendance system.

Loads the converted Teachable Machine Keras model and maps model predictions
back to labels from models/metadata.json.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Result for one recognized face image."""

    person_id: str
    confidence: float
    class_index: int
    scores: List[float]
    threshold: float

    @property
    def is_unknown(self) -> bool:
        return self.person_id.strip().lower() == "unknown"

    @property
    def is_above_threshold(self) -> bool:
        return self.confidence >= self.threshold


class FaceRecognizer:
    """Keras-backed recognizer for Teachable Machine image models."""

    MODEL_INPUT_SIZE = (224, 224)

    def __init__(self, model_path: Path, metadata_path: Path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.confidence_threshold = 0.85

        self.labels = self._load_labels()
        self.model = self._load_model()
        self._validate_model_output()

    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence required before attendance is marked."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold

    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize a single aligned RGB face image."""
        return self.recognize_batch([face_image])[0]

    def recognize_batch(
        self, face_images: Sequence[np.ndarray]
    ) -> List[RecognitionResult]:
        """Recognize multiple aligned RGB face images in one model call."""
        if not face_images:
            return []

        batch = np.stack([self._preprocess_image(img) for img in face_images], axis=0)
        predictions = self.model.predict(batch, verbose=0)

        results = []
        for scores in np.asarray(predictions):
            class_index = int(np.argmax(scores))
            confidence = float(scores[class_index])
            person_id = self.labels[class_index] if class_index < len(self.labels) else "Unknown"
            results.append(
                RecognitionResult(
                    person_id=person_id,
                    confidence=confidence,
                    class_index=class_index,
                    scores=[float(score) for score in scores],
                    threshold=self.confidence_threshold,
                )
            )

        return results

    def _load_labels(self) -> List[str]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        labels = metadata.get("labels")
        if not labels:
            raise ValueError(f"No labels found in metadata file: {self.metadata_path}")

        return [str(label).strip() for label in labels]

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"converted model not found: {self.model_path}")

        try:
            import tf_keras
        except ImportError:
            tf_keras = None

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed in this Python environment. "
                "Use the Python 3.12 virtualenv or install dependencies with "
                "pip install -r requirements.txt."
            ) from exc

        loader = tf_keras.models if tf_keras is not None else tf.keras.models
        model = loader.load_model(str(self.model_path))
        logger.info("Recognition model loaded from %s", self.model_path)
        return model

    def _validate_model_output(self):
        output_shape = getattr(self.model, "output_shape", None)
        if not output_shape:
            return

        output_classes = output_shape[-1]
        if output_classes is not None and int(output_classes) != len(self.labels):
            raise ValueError(
                f"Model outputs {output_classes} classes, but metadata has "
                f"{len(self.labels)} labels: {self.labels}"
            )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Cannot recognize an empty face image")

        import cv2

        if image.shape[:2] != self.MODEL_INPUT_SIZE:
            image = cv2.resize(image, self.MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)

        image = image.astype(np.float32)
        return (image / 127.5) - 1.0
