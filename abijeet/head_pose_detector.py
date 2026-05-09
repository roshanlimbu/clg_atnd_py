"""
========================================================
head_pose_detector.py — Head Pose Estimation Module
========================================================
Detects head orientation (front, left profile, right profile, up, down)
using MediaPipe facial landmarks and 3D geometry.

This enables the system to:
- Categorize face embeddings by pose angle
- Recognize people from any angle (profile faces)
- Build multi-angle enrollment data during collection mode
========================================================
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class HeadPose(Enum):
    """Head pose classification."""
    FRONT = "front"           # -15° to 15°
    LEFT = "left"             # -90° to -30°
    RIGHT = "right"           # 30° to 90°
    UP = "up"                 # > 15° (pitch up)
    DOWN = "down"             # < -15° (pitch down)
    LEFT_PROFILE = "left_profile"    # -90° to -45°
    RIGHT_PROFILE = "right_profile"  # 45° to 90°


@dataclass
class PoseEstimate:
    """Head pose estimation result."""
    yaw: float      # Left (-) / Right (+) in degrees [-90, 90]
    pitch: float    # Down (-) / Up (+) in degrees [-90, 90]
    roll: float     # Counterclockwise (-) / Clockwise (+) in degrees [-90, 90]
    pose_label: HeadPose


class HeadPoseDetector:
    """
    Estimates head pose (yaw, pitch, roll) using MediaPipe face landmarks
    and 3D-to-2D projection geometry.
    """

    def __init__(self):
        """Initialize MediaPipe Face Detection."""
        if mp is None:
            raise ImportError(
                "MediaPipe is required for head pose detection. "
                "Install: pip install mediapipe"
            )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 3D model points of a generic face (in mm, roughly in neutral position)
        self.model_points_3d = self._get_3d_face_model()
        logger.info("HeadPoseDetector initialized with MediaPipe Face Mesh")

    @staticmethod
    def _get_3d_face_model() -> np.ndarray:
        """
        3D coordinates of face landmarks (generic model).
        These are normalized from MediaPipe landmarks.
        """
        # Key landmarks for PnP (Perspective-n-Point) algorithm
        return np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [0.0, -330.0, -65.0],       # Chin
            [-225.0, 170.0, -135.0],    # Left eye left corner
            [225.0, 170.0, -135.0],     # Right eye right corner
            [-150.0, -150.0, -125.0],   # Left mouth corner
            [150.0, -150.0, -125.0],    # Right mouth corner
        ], dtype=np.float32)

    def estimate_pose(self, face_image: np.ndarray) -> Optional[PoseEstimate]:
        """
        Estimate head pose from a face image.

        Args:
            face_image: RGB face image (pre-cropped)

        Returns:
            PoseEstimate with yaw, pitch, roll and pose_label, or None on error
        """
        if face_image is None or face_image.size == 0:
            return None

        try:
            # Get face landmarks
            results = self.face_mesh.process(face_image)
            if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            image_h, image_w = face_image.shape[:2]

            # Extract key landmark indices
            # MediaPipe landmarks: https://mediapipe.dev/solutions/face_mesh
            landmark_indices = {
                "nose": 1,              # Nose tip
                "chin": 152,            # Chin
                "left_eye": 33,         # Left eye inner corner
                "right_eye": 263,       # Right eye inner corner
                "mouth_left": 61,       # Left mouth corner
                "mouth_right": 291,     # Right mouth corner
            }

            # Extract 2D landmarks
            image_points = np.array([
                [landmarks[landmark_indices["nose"]].x * image_w,
                 landmarks[landmark_indices["nose"]].y * image_h],
                [landmarks[landmark_indices["chin"]].x * image_w,
                 landmarks[landmark_indices["chin"]].y * image_h],
                [landmarks[landmark_indices["left_eye"]].x * image_w,
                 landmarks[landmark_indices["left_eye"]].y * image_h],
                [landmarks[landmark_indices["right_eye"]].x * image_w,
                 landmarks[landmark_indices["right_eye"]].y * image_h],
                [landmarks[landmark_indices["mouth_left"]].x * image_w,
                 landmarks[landmark_indices["mouth_left"]].y * image_h],
                [landmarks[landmark_indices["mouth_right"]].x * image_w,
                 landmarks[landmark_indices["mouth_right"]].y * image_h],
            ], dtype=np.float32)

            # Camera intrinsics (assuming calibrated camera)
            focal_length = image_w
            center = (image_w // 2, image_h // 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))

            # Solve PnP (Perspective-n-Point)
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points_3d,
                image_points,
                camera_matrix,
                dist_coeffs,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

            # Extract Euler angles (yaw, pitch, roll)
            yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)

            # Classify pose
            pose_label = self._classify_pose(yaw, pitch, roll)

            return PoseEstimate(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                pose_label=pose_label
            )

        except Exception as e:
            logger.debug(f"Pose estimation failed: {e}")
            return None

    @staticmethod
    def _rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract yaw, pitch, roll from rotation matrix using ZYX Euler angle convention.

        Returns:
            (yaw, pitch, roll) in degrees
        """
        # Clamp to avoid numerical errors in arcsin
        pitch = np.arcsin(-rotation_matrix[2, 0])

        # Avoid gimbal lock
        if np.abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            yaw = 0
            roll = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)

        return yaw, pitch, roll

    @staticmethod
    def _classify_pose(yaw: float, pitch: float, roll: float) -> HeadPose:
        """
        Classify head pose into discrete categories.

        Args:
            yaw: Left-right rotation in degrees [-90, 90]
            pitch: Up-down rotation in degrees [-90, 90]
            roll: Clockwise rotation in degrees [-90, 90]

        Returns:
            HeadPose enum value
        """
        # Check extreme profiles first
        if yaw < -45:
            return HeadPose.LEFT_PROFILE
        elif yaw > 45:
            return HeadPose.RIGHT_PROFILE
        elif -45 <= yaw < -15:
            return HeadPose.LEFT
        elif 15 < yaw <= 45:
            return HeadPose.RIGHT
        elif pitch > 15:
            return HeadPose.UP
        elif pitch < -15:
            return HeadPose.DOWN
        else:
            return HeadPose.FRONT

    def draw_pose_on_frame(
        self,
        frame: np.ndarray,
        pose: PoseEstimate,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw head pose estimation on frame (for debugging).

        Args:
            frame: Input frame
            pose: Pose estimation result
            bbox: Bounding box (x1, y1, x2, y2) for position

        Returns:
            Frame with drawn pose info
        """
        if bbox is None:
            h, w = frame.shape[:2]
            bbox = (10, 10, w - 10, 50)

        x1, y1, x2, y2 = bbox
        
        # Draw background
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        # Draw pose text
        pose_text = (
            f"Pose: {pose.pose_label.value} | "
            f"Yaw: {pose.yaw:.1f}° | "
            f"Pitch: {pose.pitch:.1f}° | "
            f"Roll: {pose.roll:.1f}°"
        )
        
        cv2.putText(
            frame,
            pose_text,
            (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return frame


# Thresholds for pose-based matching adjustments
POSE_DISTANCE_ADJUSTMENTS = {
    # (source_pose, target_pose): adjustment_factor
    (HeadPose.FRONT, HeadPose.FRONT): 1.0,           # Identical pose
    (HeadPose.FRONT, HeadPose.LEFT): 1.05,           # Small angle difference
    (HeadPose.FRONT, HeadPose.RIGHT): 1.05,          # Small angle difference
    (HeadPose.FRONT, HeadPose.LEFT_PROFILE): 1.15,   # Large angle difference
    (HeadPose.FRONT, HeadPose.RIGHT_PROFILE): 1.15,  # Large angle difference
    (HeadPose.LEFT, HeadPose.LEFT_PROFILE): 1.10,    # Moderate difference
    (HeadPose.RIGHT, HeadPose.RIGHT_PROFILE): 1.10,  # Moderate difference
    (HeadPose.LEFT, HeadPose.RIGHT): 1.25,           # Opposite sides
}


def get_pose_adjusted_threshold(
    source_pose: HeadPose,
    target_pose: HeadPose,
    base_threshold: float = 0.60
) -> float:
    """
    Adjust matching threshold based on head pose difference.

    More different poses = higher threshold (more lenient matching).

    Args:
        source_pose: Pose of reference embedding
        target_pose: Pose of face being matched
        base_threshold: Base L2 distance threshold

    Returns:
        Adjusted threshold
    """
    key = (source_pose, target_pose)
    if key not in POSE_DISTANCE_ADJUSTMENTS:
        # Default to higher threshold for unknown combinations
        adjustment = 1.2
    else:
        adjustment = POSE_DISTANCE_ADJUSTMENTS[key]

    adjusted = base_threshold * adjustment
    return min(adjusted, 0.75)  # Cap at 0.75 to avoid too-loose matching
