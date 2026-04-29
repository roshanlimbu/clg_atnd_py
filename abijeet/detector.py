"""
========================================================
detector.py — Face Detection & Alignment Module
========================================================
STEP 8 + STEP 9

Two-stage pipeline for each sampled frame:

Stage 1 — YOLOv8n Face Detection:
    - Receives a full live frame from the camera
    - Detects ALL faces in a single pass (handles 20-30+ faces)
    - Returns bounding boxes for every detected face
    - Runs on CPU — no GPU required

Stage 2 — MediaPipe Face Alignment:
    - For each bounding box from YOLO
    - Crops the face region from the frame
    - Uses MediaPipe landmarks to align the face
    - Normalizes and resizes to 224x224 for model input
    - Significantly improves recognition accuracy for tilted/angled faces
========================================================
"""

import cv2
import os
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """
    Holds all data for a single detected and aligned face.
    Passed from detector → recognizer → attendance recorder.
    """
    # Bounding box in the original frame (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int]

    # Aligned and resized face image ready for model input (224x224x3)
    aligned_image: np.ndarray

    # Detection confidence from YOLOv8 (0.0 to 1.0)
    detection_confidence: float

    # Center point of the face in the original frame
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


class FaceDetector:
    """
    STEP 8 + STEP 9 implementation.

    Combines YOLOv8n (fast multi-face detection) with
    MediaPipe (face alignment/normalization) into one pipeline.
    """

    # Target size for Teachable Machine model input
    MODEL_INPUT_SIZE = (224, 224)

    # Minimum face size to process (pixels) — filters out tiny far-away faces
    MIN_FACE_SIZE = 60

    # OpenCV Haar is only a fallback. The embedding stage now rejects unusable
    # crops, so favor recall here instead of requiring perfect Haar detections.
    OPENCV_MIN_NEIGHBORS = 5
    MIN_SKIN_RATIO = 0.08

    # Padding around detected face before alignment (fraction of face size)
    FACE_PADDING = 0.2

    # This environment's MediaPipe package exposes only the newer tasks API,
    # not mp.solutions.face_mesh. Keep alignment off unless that API is present.
    ENABLE_MEDIAPIPE_ALIGNMENT = False

    def __init__(self):
        """Load YOLOv8n and MediaPipe models."""
        self._yolo_model = None
        self._face_recognition = None
        self._opencv_face_cascade = None
        self._opencv_eye_cascade = None
        self._mp_face_mesh = None
        self._mp_drawing = None
        self._cache_dir = Path(__file__).parent / ".cache"

        self._load_yolo()
        self._load_face_recognition_detector()
        self._load_mediapipe()

    def _load_yolo(self):
        """
        STEP 8 — Load YOLOv8n face detection model.

        Uses the 'yolov8n-face' variant (nano size, face-specific weights).
        If face-specific weights aren't available, falls back to OpenCV's
        face cascade. Do not use general yolov8n.pt here: that model detects
        arbitrary objects and creates false attendance entries.

        First run: ultralytics auto-downloads the model weights.
        """
        yolo_cache_dir = self._cache_dir / "ultralytics"
        yolo_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_cache_dir))

        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.warning(f"Ultralytics not available, using OpenCV face detector: {e}")
            self._load_opencv_face_detector()
            return

        try:
            self._yolo_model = YOLO("yolov8n-face.pt")
            logger.info("YOLOv8n-face model loaded (face-specific weights)")
            return
        except Exception as e:
            logger.warning(f"Could not load yolov8n-face.pt: {e}")
            logger.info("Using OpenCV face detector fallback")
            self._load_opencv_face_detector()

    def _load_face_recognition_detector(self):
        """Load dlib HOG face detector via face_recognition as a recall fallback."""
        try:
            import face_recognition as fr
            self._face_recognition = fr
            logger.info("dlib HOG face detector loaded via face_recognition")
        except Exception as e:
            logger.warning("face_recognition detector unavailable: %s", e)
            self._face_recognition = None

    def _load_opencv_face_detector(self):
        """Load OpenCV's bundled Haar face cascade as an offline fallback."""
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        eye_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
        self._opencv_face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self._opencv_eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
        if self._opencv_face_cascade.empty() or self._opencv_eye_cascade.empty():
            raise RuntimeError(
                "Could not load YOLO or OpenCV Haar face/eye detector. "
                "Install ultralytics or check your OpenCV installation."
            )
        logger.info("OpenCV Haar face detector loaded with validation filters")

    def _load_mediapipe(self):
        """
        STEP 9 — Initialize MediaPipe Face Mesh for alignment.

        Face Mesh detects 468 facial landmarks (eyes, nose, mouth corners, etc.)
        Used to compute the face's rotation angle and straighten it.
        """
        if not self.ENABLE_MEDIAPIPE_ALIGNMENT:
            logger.info("MediaPipe alignment disabled; using direct face crops")
            self._mp_face_mesh = None
            return

        matplotlib_cache_dir = self._cache_dir / "matplotlib"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                logger.warning(
                    "MediaPipe %s does not expose mp.solutions; "
                    "continuing without landmark alignment",
                    getattr(mp, "__version__", "unknown"),
                )
                self._mp_face_mesh = None
                return

            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,     # Each crop is treated as a static image
                max_num_faces=1,            # Only one face per crop at a time
                refine_landmarks=True,      # More accurate eye/lip landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe FaceMesh initialized")

        except Exception as e:
            logger.warning(
                "Failed to initialize MediaPipe alignment; "
                "continuing with direct face crops: %s",
                e,
            )
            self._mp_face_mesh = None

    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        STEP 8 + STEP 9 — Main pipeline: detect all faces and align each one.

        Args:
            frame: Full BGR frame from the camera (OpenCV format)

        Returns:
            List of DetectedFace objects, one per detected face.
            Empty list if no faces found.
        """
        if frame is None or frame.size == 0:
            return []

        # Stage 1: detect all face bounding boxes. Use every available
        # detector, then merge overlaps. This keeps recall high when the
        # face-specific YOLO weights are unavailable.
        bounding_boxes = []
        bounding_boxes.extend(self._run_yolo_detection(frame))
        bounding_boxes.extend(self._run_dlib_detection(frame))
        bounding_boxes.extend(self._run_opencv_detection(frame))
        bounding_boxes = self._merge_overlapping_boxes(bounding_boxes)

        if not bounding_boxes:
            return []

        # Stage 2: MediaPipe — align each detected face
        detected_faces = []
        for bbox, det_confidence in bounding_boxes:
            aligned = self._align_face(frame, bbox)
            if aligned is not None:
                detected_faces.append(DetectedFace(
                    bbox=bbox,
                    aligned_image=aligned,
                    detection_confidence=det_confidence,
                ))

        logger.debug(
            f"Detected {len(detected_faces)} faces "
            f"(from {len(bounding_boxes)} merged detections)"
        )
        return detected_faces

    def _run_yolo_detection(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Run YOLOv8n on the full frame and return all face bounding boxes.

        YOLOv8 processes the entire frame in one forward pass —
        this is what makes it efficient for 20-30+ faces simultaneously.

        Returns:
            List of ((x1, y1, x2, y2), confidence) tuples
        """
        if self._yolo_model is None:
            return []

        try:
            results = self._yolo_model(
                frame,
                verbose=False,      # Suppress YOLO's console output
                conf=0.4,           # Minimum detection confidence threshold
                iou=0.5,            # IOU threshold for non-max suppression
            )
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

        bboxes = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())

                # Filter out tiny detections (far-away faces, false positives)
                width = x2 - x1
                height = y2 - y1
                if width < self.MIN_FACE_SIZE or height < self.MIN_FACE_SIZE:
                    continue

                # Clamp to frame boundaries
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                bboxes.append(((x1, y1, x2, y2), confidence))

        return bboxes

    def _run_dlib_detection(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Run dlib's HOG face detector through face_recognition."""
        if self._face_recognition is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            locations = self._face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=1,
                model="hog",
            )
        except Exception as e:
            logger.debug("dlib face detection failed: %s", e)
            return []

        h, w = frame.shape[:2]
        bboxes = []
        for top, right, bottom, left in locations:
            x1 = max(0, int(left))
            y1 = max(0, int(top))
            x2 = min(w, int(right))
            y2 = min(h, int(bottom))
            if x2 - x1 < self.MIN_FACE_SIZE or y2 - y1 < self.MIN_FACE_SIZE:
                continue
            bboxes.append(((x1, y1, x2, y2), 0.95))

        return bboxes

    def _run_opencv_detection(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Run the OpenCV Haar fallback detector."""
        if self._opencv_face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._opencv_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=self.OPENCV_MIN_NEIGHBORS,
            minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        bboxes = []
        for x, y, w, h in faces:
            bbox = (int(x), int(y), int(x + w), int(y + h))
            if self._is_valid_opencv_face(frame, gray, bbox):
                bboxes.append((bbox, 1.0))

        return bboxes

    def _is_valid_opencv_face(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """Reject common Haar false positives before they reach attendance."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False

        aspect_ratio = width / height
        if aspect_ratio < 0.55 or aspect_ratio > 1.35:
            return False

        face_gray = gray[y1:y2, x1:x2]
        face_bgr = frame[y1:y2, x1:x2]
        if face_gray.size == 0 or face_bgr.size == 0:
            return False

        upper_half = face_gray[: max(1, height // 2), :]
        min_eye_size = max(10, int(min(width, height) * 0.12))
        eyes = self._opencv_eye_cascade.detectMultiScale(
            upper_half,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_eye_size, min_eye_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        skin_ok = self._skin_ratio(face_bgr) >= self.MIN_SKIN_RATIO
        return skin_ok or len(eyes) >= 1

    def _merge_overlapping_boxes(
        self,
        boxes: List[Tuple[Tuple[int, int, int, int], float]],
        iou_threshold: float = 0.35,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Merge duplicate detections from YOLO, dlib, and OpenCV."""
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda item: item[1], reverse=True)
        merged: List[Tuple[Tuple[int, int, int, int], float]] = []
        for bbox, confidence in sorted_boxes:
            if any(self._iou(bbox, existing_bbox) > iou_threshold for existing_bbox, _ in merged):
                continue
            merged.append((bbox, confidence))

        return sorted(merged, key=lambda item: item[0][0])

    @staticmethod
    def _iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        """Intersection-over-union for two xyxy boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _skin_ratio(self, face_bgr: np.ndarray) -> float:
        """Estimate whether a crop contains enough skin-like pixels."""
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        skin_mask = (
            (cr >= 133) & (cr <= 180) &
            (cb >= 70) & (cb <= 135)
        )
        return float(np.count_nonzero(skin_mask)) / float(skin_mask.size)

    def _align_face(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        STEP 9 — Crop, align, and normalize a single face.

        Process:
        1. Crop face from frame with padding
        2. Convert to RGB for MediaPipe
        3. Detect facial landmarks with MediaPipe Face Mesh
        4. Compute rotation angle from eye positions
        5. Rotate/align the face to be upright
        6. Resize to 224x224 for the Keras model

        Args:
            frame: Full BGR frame
            bbox: (x1, y1, x2, y2) bounding box from YOLO

        Returns:
            Aligned RGB image as numpy array (224, 224, 3)
            or None if alignment fails (MediaPipe couldn't find landmarks)
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]

        # Add padding around the face crop for better landmark detection
        pad_w = int((x2 - x1) * self.FACE_PADDING)
        pad_h = int((y2 - y1) * self.FACE_PADDING)

        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w_frame, x2 + pad_w)
        crop_y2 = min(h_frame, y2 + pad_h)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if face_crop.size == 0:
            return None

        # Convert BGR → RGB for MediaPipe
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        if self._mp_face_mesh is None:
            return self._direct_resize(face_rgb)

        # Run MediaPipe landmark detection
        try:
            results = self._mp_face_mesh.process(face_rgb)
        except Exception as e:
            logger.debug(f"MediaPipe processing error: {e}")
            # Fall back to direct resize without alignment
            return self._direct_resize(face_rgb)

        if not results.multi_face_landmarks:
            # MediaPipe couldn't find landmarks — use direct resize as fallback
            logger.debug("MediaPipe found no landmarks — using direct resize")
            return self._direct_resize(face_rgb)

        # Get the first (only) face's landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        h_crop, w_crop = face_crop.shape[:2]

        # ── Compute eye center positions for rotation correction ──────────
        # Left eye: landmarks 33 (outer corner), 133 (inner corner)
        # Right eye: landmarks 362 (outer corner), 263 (inner corner)
        # Using outer corners for stable alignment reference

        left_eye_x = int(landmarks[33].x * w_crop)
        left_eye_y = int(landmarks[33].y * h_crop)
        right_eye_x = int(landmarks[362].x * w_crop)
        right_eye_y = int(landmarks[362].y * h_crop)

        # Compute the angle between eyes
        dy = right_eye_y - left_eye_y
        dx = right_eye_x - left_eye_x
        angle = np.degrees(np.arctan2(dy, dx))

        # Only rotate if the tilt is significant (> 2 degrees)
        if abs(angle) > 2.0:
            # Rotate around the midpoint between the eyes
            eye_center = (
                (left_eye_x + right_eye_x) // 2,
                (left_eye_y + right_eye_y) // 2,
            )
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            aligned_face = cv2.warpAffine(
                face_rgb, rotation_matrix, (w_crop, h_crop),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            aligned_face = face_rgb

        return self._direct_resize(aligned_face)

    def _direct_resize(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Resize face image to 224x224 for Teachable Machine model input.
        Uses LANCZOS for high-quality downsampling.
        """
        return cv2.resize(
            face_rgb,
            self.MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LANCZOS4,
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        detected_faces: List[DetectedFace],
        recognition_results: Optional[List[dict]] = None,
    ) -> np.ndarray:
        """
        STEP 14 — Draw bounding boxes and labels on the live frame.

        Color coding:
            Green  (#00FF00) — Successfully marked present
            Yellow (#FFFF00) — Already marked today
            Red    (#FF0000) — Unknown face
            Gray   (#888888) — Below confidence threshold

        Args:
            frame             : Original BGR frame from camera
            detected_faces    : List of DetectedFace objects
            recognition_results: Optional list of dicts with recognition data
                                 Keys: person_id, confidence, status
                                 status: 'marked' | 'already_marked' | 'unknown' | 'low_confidence'

        Returns:
            Annotated frame with bounding boxes and labels drawn
        """
        annotated = frame.copy()

        # Color map for each detection status
        COLOR_MAP = {
            "marked":           (0,   255,  0),    # Green
            "already_marked":   (0,   255, 255),   # Yellow
            "unknown":          (0,   0,   255),   # Red
            "low_confidence":   (128, 128, 128),   # Gray
        }

        LABEL_MAP = {
            "marked":           "Counted",
            "already_marked":   "Debounced",
            "unknown":          "Unknown",
            "low_confidence":   "Low Confidence",
        }

        for i, face in enumerate(detected_faces):
            x1, y1, x2, y2 = face.bbox

            # Get recognition result if available
            result = None
            if recognition_results and i < len(recognition_results):
                result = recognition_results[i]

            status = result.get("status", "low_confidence") if result else "low_confidence"
            color = COLOR_MAP.get(status, (128, 128, 128))

            # Draw bounding box
            thickness = 2 if status == "marked" else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            if result:
                person_id = result.get("person_id", "?")
                conf = result.get("confidence", 0.0)
                count = result.get("count", 0)
                status_label = LABEL_MAP.get(status, status)

                if status in ("unknown", "low_confidence"):
                    label = f"{status_label} ({conf:.0%})"
                else:
                    label = f"{person_id} | {status_label} #{count}"
            else:
                label = "Processing..."

            # Draw label background for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            label_y = max(y1 - 5, text_h + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - baseline - 2),
                (x1 + text_w + 4, label_y + baseline),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated, label,
                (x1 + 2, label_y - baseline),
                font, font_scale, (0, 0, 0), font_thickness,
                cv2.LINE_AA,
            )

        return annotated

    def release(self):
        """Release MediaPipe resources."""
        if self._mp_face_mesh:
            self._mp_face_mesh.close()
            logger.info("MediaPipe FaceMesh released")
