"""
========================================================
detector.py — Face Detection & Alignment Module
========================================================
STEP 8 + STEP 9

Three-stage pipeline for each sampled frame:

Stage 1 — YOLOv8 Face Detection (with sandwich-face fixes):
    - Splits frame into overlapping tiles to prevent NMS suppression
    - Runs multi-scale detection (100% + 150%) per tile
    - Uses lowered NMS IOU threshold (0.25) to keep occluded faces
    - Merges and deduplicates all detections
    - Runs on CPU — no GPU required

Stage 2 — MediaPipe Face Alignment:
    - For each bounding box from YOLO
    - Crops the face region from the frame
    - Uses MediaPipe landmarks to align the face
    - Normalizes and resizes to 224x224 for model input
    - Significantly improves recognition accuracy for tilted/angled faces

Sandwich Face Problem:
    When a face sits between two closer people, NMS would remove it
    thinking it overlaps with the larger surrounding boxes. The tiling
    approach isolates each region so the back face becomes dominant in
    its own tile, and the lowered IOU threshold prevents aggressive
    suppression of legitimate detections.
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

# ArcFace standard reference landmarks for 112x112 aligned output.
# These are the canonical positions ArcFace was trained on.
# Source: insightface/utils/face_align.py
ARCFACE_REF_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


@dataclass
class DetectedFace:
    """
    Holds all data for a single detected and aligned face.
    Passed from detector -> recognizer -> attendance recorder.
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
    """YOLOv8 face detection + optional MediaPipe alignment.

    Includes sandwich-face recovery via tiled + multi-scale detection
    with relaxed NMS thresholds.
    """

    # Target size for aligned face output (ArcFace native = 112x112)
    MODEL_INPUT_SIZE = (112, 112)

    # Minimum face size to process (pixels)
    MIN_FACE_SIZE = 30

    # Padding around detected face for landmark detection (fraction of face)
    FACE_PADDING = 0.3

    # MediaPipe alignment: provides 468 landmarks for affine alignment
    ENABLE_MEDIAPIPE_ALIGNMENT = True

    # Edge enhancement: unsharp masking to sharpen face boundaries
    ENABLE_EDGE_ENHANCEMENT = True
    UNSHARP_SIGMA = 2.0     # Gaussian sigma (medium radius for feature edges)
    UNSHARP_STRENGTH = 0.5  # Enhancement strength (0.3–0.7 recommended)

    # ── Sandwich-face fix parameters ──────────────────────────────────
    # NMS IOU threshold: lower = less aggressive suppression.
    # Default YOLO uses 0.45–0.5; we use 0.25 to keep sandwich faces.
    NMS_IOU_THRESHOLD = 0.25

    # Detection confidence threshold (0.25 keeps most real faces)
    NMS_CONF_THRESHOLD = 0.25

    # Enable frame tiling to isolate sandwich faces (Fix 3)
    # WARNING: Each tile adds a full YOLO call. On CPU this tanks FPS.
    # Enable only on GPU or when accuracy matters more than speed.
    ENABLE_TILING = False

    # Number of tiles to split the frame into (horizontal)
    TILE_COLS = 2
    TILE_ROWS = 1  # 1 row for typical crowd scenes (faces at similar height)

    # Overlap between adjacent tiles (fraction of tile size)
    TILE_OVERLAP = 0.20

    # Enable multi-scale detection (Fix 4)
    # WARNING: Doubles YOLO calls per tile/frame. Heavy on CPU.
    # Enable only on GPU or when small distant faces are missed.
    ENABLE_MULTI_SCALE = False

    # Scale factors for multi-scale detection
    SCALE_FACTORS = [1.0, 1.5]

    # IOU threshold for merging duplicate detections across tiles/scales
    MERGE_IOU_THRESHOLD = 0.60

    # Maximum faces to process per frame (largest/closest first).
    # In dense crowds, processing 30+ faces tanks FPS. Cap at 10-12
    # for walking-crowd scenarios where only nearby faces matter.
    MAX_FACES_PER_FRAME = 10

    def __init__(self):
        """Load YOLOv8n and InsightFace landmark models."""
        self._yolo_model = None
        self._landmark_model = None  # InsightFace detection for landmarks
        self._cache_dir = Path(__file__).parent / ".cache"

        self._load_yolo()
        self._load_landmark_detector()

    def _load_yolo(self):
        """Load YOLOv8 model. Raises if unavailable -- no fallback detectors."""
        yolo_cache_dir = self._cache_dir / "ultralytics"
        yolo_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_cache_dir))

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics is required for face detection. "
                "Install it with: pip install ultralytics"
            ) from e

        # Look for model in models/ directory first, then root
        base_dir = Path(__file__).parent
        model_path = base_dir / "models" / "yolov8n.pt"
        if not model_path.exists():
            model_path = base_dir / "yolov8n.pt"  # fallback

        self._yolo_model = YOLO(str(model_path))
        logger.info("YOLOv8n model loaded from %s", model_path.name)

    def _load_landmark_detector(self):
        """
        STEP 9 — Load InsightFace detection model for 5-point landmarks.

        Uses the same InsightFace package already loaded for recognition.
        The detection model provides 5-point landmarks (eyes, nose, mouth)
        which are used for ArcFace affine alignment.
        """
        if not self.ENABLE_MEDIAPIPE_ALIGNMENT:
            logger.info("Landmark alignment disabled; using direct face crops")
            self._landmark_model = None
            return

        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],  # Only detection, no recognition
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(160, 160))  # Small size for speed
            self._landmark_model = app
            logger.info("InsightFace landmark detector loaded (for affine alignment)")

        except Exception as e:
            logger.warning(
                "Failed to load InsightFace landmark detector; "
                "continuing with direct face crops: %s", e,
            )
            self._landmark_model = None

    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        STEP 8 + STEP 9 -- Detect faces with enhanced sandwich-face pipeline.

        Pipeline:
        1. Split frame into overlapping tiles (if ENABLE_TILING)
        2. Run YOLO on each tile at multiple scales (if ENABLE_MULTI_SCALE)
        3. Convert tile-local coords back to full-frame coords
        4. Merge and deduplicate all detections
        5. Align each surviving detection with MediaPipe

        Args:
            frame: Full BGR frame from the camera (OpenCV format)

        Returns:
            List of DetectedFace objects, one per detected face.
            Empty list if no faces found.
        """
        if frame is None or frame.size == 0:
            return []

        # ── Enhanced detection with tiling + multi-scale ──────────────
        if self.ENABLE_TILING:
            bounding_boxes = self._run_tiled_detection(frame)
        else:
            bounding_boxes = self._run_detection_multi_scale(frame)

        if not bounding_boxes:
            return []

        # ── Sort by face area (largest = closest) and cap count ──────
        bounding_boxes.sort(
            key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]),
            reverse=True,
        )
        if self.MAX_FACES_PER_FRAME > 0:
            bounding_boxes = bounding_boxes[:self.MAX_FACES_PER_FRAME]

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
            "Detected %d faces (from %d raw detections)",
            len(detected_faces), len(bounding_boxes),
        )
        return detected_faces

    # ── Tiled detection (Fix 3) ────────────────────────────────────────

    def _run_tiled_detection(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Split the frame into overlapping tiles, run detection on each,
        then merge results back into full-frame coordinates.

        This isolates sandwich faces: in their own tile, the back face
        becomes the dominant detection and avoids suppression from the
        larger surrounding front-face boxes.
        """
        h, w = frame.shape[:2]
        all_detections = []

        tiles = self._generate_tiles(w, h)

        for (tx1, ty1, tx2, ty2) in tiles:
            tile = frame[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            # Run detection on this tile (optionally multi-scale)
            tile_detections = self._run_detection_multi_scale(tile)

            # Convert tile-local coordinates to full-frame coordinates
            for (bx1, by1, bx2, by2), conf in tile_detections:
                full_bbox = (bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1)
                all_detections.append((full_bbox, conf))

        # Also run on the full frame to catch faces at tile boundaries
        full_detections = self._run_detection_multi_scale(frame)
        all_detections.extend(full_detections)

        # Deduplicate across tiles using a higher IOU threshold
        merged = self._deduplicate_detections(all_detections, self.MERGE_IOU_THRESHOLD)

        logger.debug(
            "Tiled detection: %d tiles × %d scales → %d raw → %d merged",
            len(tiles), len(self.SCALE_FACTORS) if self.ENABLE_MULTI_SCALE else 1,
            len(all_detections), len(merged),
        )
        return merged

    def _generate_tiles(
        self, frame_w: int, frame_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate overlapping tile coordinates.

        Returns:
            List of (x1, y1, x2, y2) tuples for each tile.
        """
        tiles = []
        tile_w = frame_w // self.TILE_COLS
        tile_h = frame_h // self.TILE_ROWS
        overlap_w = int(tile_w * self.TILE_OVERLAP)
        overlap_h = int(tile_h * self.TILE_OVERLAP)

        for row in range(self.TILE_ROWS):
            for col in range(self.TILE_COLS):
                x1 = max(0, col * tile_w - overlap_w)
                y1 = max(0, row * tile_h - overlap_h)
                x2 = min(frame_w, (col + 1) * tile_w + overlap_w)
                y2 = min(frame_h, (row + 1) * tile_h + overlap_h)
                tiles.append((x1, y1, x2, y2))

        return tiles

    # ── Multi-scale detection (Fix 4) ─────────────────────────────────

    def _run_detection_multi_scale(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Run detection at multiple scales and merge results.
        At larger scales, small distant back-faces become large enough
        to be reliably detected.
        """
        if not self.ENABLE_MULTI_SCALE or len(self.SCALE_FACTORS) <= 1:
            return self._run_yolo_detection(frame)

        h, w = frame.shape[:2]
        all_detections = []

        for scale in self.SCALE_FACTORS:
            if scale == 1.0:
                scaled_frame = frame
            else:
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_frame = cv2.resize(
                    frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

            detections = self._run_yolo_detection(scaled_frame)

            # Scale coordinates back to original frame size
            if scale != 1.0:
                for (bx1, by1, bx2, by2), conf in detections:
                    orig_bbox = (
                        int(bx1 / scale),
                        int(by1 / scale),
                        int(bx2 / scale),
                        int(by2 / scale),
                    )
                    all_detections.append((orig_bbox, conf))
            else:
                all_detections.extend(detections)

        # Deduplicate across scales
        return self._deduplicate_detections(all_detections, self.MERGE_IOU_THRESHOLD)

    # ── Core YOLO detection ───────────────────────────────────────────

    def _run_yolo_detection(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Run YOLOv8 on a single frame/tile and return all bounding boxes.

        Uses relaxed NMS thresholds (IOU=0.25) to prevent suppression
        of sandwich faces.

        Returns:
            List of ((x1, y1, x2, y2), confidence) tuples
        """
        if self._yolo_model is None:
            return []

        try:
            results = self._yolo_model(
                frame,
                verbose=False,
                conf=self.NMS_CONF_THRESHOLD,
                iou=self.NMS_IOU_THRESHOLD,
            )
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

        bboxes = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if hasattr(box, 'cls') and box.cls is not None:
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id != 0:
                        continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())

                width = x2 - x1
                height = y2 - y1
                if width < self.MIN_FACE_SIZE or height < self.MIN_FACE_SIZE:
                    continue

                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                bboxes.append(((x1, y1, x2, y2), confidence))

        return bboxes

    # ── Deduplication via NMS merge ────────────────────────────────────

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    @classmethod
    def _deduplicate_detections(
        cls,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        iou_threshold: float,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Merge duplicate detections from tiling/multi-scale.

        Uses a softer NMS: for truly overlapping boxes (IOU > threshold),
        keep the higher-confidence one. This is separate from YOLO's
        internal NMS — it only removes *true* duplicates created by
        overlapping tiles or scales.
        """
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d[1], reverse=True)
        keep = []

        for bbox, conf in sorted_dets:
            is_duplicate = False
            for kept_bbox, _ in keep:
                if cls._compute_iou(bbox, kept_bbox) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append((bbox, conf))

        return keep

    def _align_face(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        STEP 9 — Landmark-based affine alignment.

        Instead of cropping the YOLO bounding box (which includes neck,
        hair, and background), this uses facial landmarks to compute an
        affine transform that warps the face to ArcFace's standard
        112x112 template position.

        Pipeline:
        1. Crop generous region around YOLO bbox for landmark detection
        2. Detect 468 facial landmarks with MediaPipe Face Mesh
        3. Extract 5-point landmarks (eyes, nose, mouth corners)
        4. Compute affine transform to ArcFace reference template
        5. Warp full frame → 112x112 aligned crop (no background bleed)
        6. Apply edge enhancement (unsharp masking)

        Args:
            frame: Full BGR frame from camera
            bbox: (x1, y1, x2, y2) bounding box from YOLO

        Returns:
            Aligned RGB image (112, 112, 3) or None if alignment fails
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]

        # Generous padding for reliable landmark detection
        pad_w = int((x2 - x1) * self.FACE_PADDING)
        pad_h = int((y2 - y1) * self.FACE_PADDING)

        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w_frame, x2 + pad_w)
        crop_y2 = min(h_frame, y2 + pad_h)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if face_crop.size == 0:
            return None

        # Convert BGR → RGB for fallback output
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        if self._landmark_model is None:
            # No landmark model — fall back to direct crop + resize
            result = self._direct_resize(face_rgb)
            return self._enhance_edges(result) if self.ENABLE_EDGE_ENHANCEMENT else result

        # ── InsightFace landmark detection on the padded crop ─────────
        try:
            faces = self._landmark_model.get(face_crop)  # BGR input
        except Exception as e:
            logger.debug("InsightFace landmark error: %s", e)
            result = self._direct_resize(face_rgb)
            return self._enhance_edges(result) if self.ENABLE_EDGE_ENHANCEMENT else result

        if not faces or faces[0].kps is None:
            logger.debug("No landmarks found — using direct resize")
            result = self._direct_resize(face_rgb)
            return self._enhance_edges(result) if self.ENABLE_EDGE_ENHANCEMENT else result

        # InsightFace returns 5-point kps in crop coordinates.
        # Map to full-frame coordinates by adding crop offset.
        kps = faces[0].kps.copy()  # shape (5, 2)
        kps[:, 0] += crop_x1
        kps[:, 1] += crop_y1

        # ── Affine warp from detected landmarks to ArcFace template ──
        aligned = self._affine_align(frame, kps.astype(np.float32))
        if aligned is None:
            result = self._direct_resize(face_rgb)
            return self._enhance_edges(result) if self.ENABLE_EDGE_ENHANCEMENT else result

        # Convert BGR → RGB (affine warp was on the BGR frame)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        return self._enhance_edges(aligned_rgb) if self.ENABLE_EDGE_ENHANCEMENT else aligned_rgb

    def _affine_align(
        self, frame_bgr: np.ndarray, src_landmarks: np.ndarray,
        output_size: int = 112,
    ) -> Optional[np.ndarray]:
        """
        Compute a similarity transform from detected 5-point landmarks
        to ArcFace's canonical reference positions and warp the face.

        This produces the exact 112x112 crop ArcFace was trained on:
        - Face centered and upright
        - Eyes at standard positions
        - Background pixels minimised
        - No bounding-box bleed

        Args:
            frame_bgr: Full BGR frame (warp uses the full image)
            src_landmarks: 5x2 array of detected landmark positions
            output_size: Output size (default 112 for ArcFace)

        Returns:
            Aligned BGR face (112x112x3) or None if transform fails
        """
        dst = ARCFACE_REF_112.copy()

        # estimateAffinePartial2D computes similarity transform
        # (rotation + uniform scale + translation — 4 DOF)
        # This preserves face proportions unlike a full affine (6 DOF).
        tform, inliers = cv2.estimateAffinePartial2D(
            src_landmarks.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )

        if tform is None:
            logger.debug("Affine transform estimation failed")
            return None

        aligned = cv2.warpAffine(
            frame_bgr, tform, (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def _enhance_edges(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Unsharp masking: sharpen facial feature edges without
        affecting smooth skin regions.

        Process: sharpened = original + strength * (original - blurred)
        Cost: ~2ms — runs on every crop with negligible FPS impact.
        """
        blurred = cv2.GaussianBlur(
            face_rgb, (0, 0), self.UNSHARP_SIGMA,
        )
        sharpened = cv2.addWeighted(
            face_rgb, 1.0 + self.UNSHARP_STRENGTH,
            blurred, -self.UNSHARP_STRENGTH,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _direct_resize(self, face_rgb: np.ndarray) -> np.ndarray:
        """Fallback: resize face to 112x112 without landmark alignment."""
        return cv2.resize(
            face_rgb,
            self.MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
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
            Blue   (#3388FF) — Internal team member, not counted
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
            "internal":         (255, 136, 51),    # Blue
            "unknown":          (0,   0,   255),   # Red
            "low_confidence":   (128, 128, 128),   # Gray
            "new_face":         (0,   200, 255),   # Orange/Gold — newly registered
        }

        LABEL_MAP = {
            "marked":           "Counted",
            "already_marked":   "Debounced",
            "internal":         "Internal",
            "unknown":          "Unknown",
            "low_confidence":   "Low Confidence",
            "new_face":         "New",
        }

        for i, face in enumerate(detected_faces):
            x1, y1, x2, y2 = face.bbox

            # Get recognition result if available
            result = None
            if recognition_results and i < len(recognition_results):
                result = recognition_results[i]

            status = result.get("status", "low_confidence") if result else "low_confidence"
            color = COLOR_MAP.get(status, (128, 128, 128))

            # Draw bounding box — thicker for counted and new faces
            thickness = 2 if status in ("marked", "new_face") else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            if result:
                person_id = result.get("person_id", "?")
                display_name = result.get("display_name") or person_id
                conf = result.get("confidence", 0.0)
                count = result.get("count", 0)
                status_label = LABEL_MAP.get(status, status)

                if status in ("unknown", "low_confidence"):
                    label = f"{status_label} ({conf:.0%})"
                elif status == "internal":
                    label = f"{display_name} | Not counted"
                elif status == "new_face":
                    label = f"{person_id} | NEW #{count}"
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
