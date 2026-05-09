"""
========================================================
enrollment_mode.py — Multi-Angle Face Enrollment
========================================================
Guided enrollment mode that helps users capture their face from multiple angles
(front, left profile, right profile, up, down) to build a comprehensive
face recognition database.

This dramatically improves recognition accuracy when people turn their heads
or approach the camera from different angles.
========================================================
"""

import logging
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class EnrollmentMode:
    """
    Interactive enrollment mode for capturing multi-angle faces.
    
    Flow:
    1. User enters enrollment mode with their name
    2. System guides them through 5 poses: front, left, right, up, down
    3. For each pose, system captures 3 frames for redundancy
    4. All captures are stored and used to initialize the face embeddings
    5. User exits and the system builds the face identity
    """

    REQUIRED_POSES = [
        "front",
        "left_profile",
        "right_profile", 
        "up",
        "down",
    ]

    FRAMES_PER_POSE = 3
    CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize enrollment mode.

        Args:
            output_dir: Directory to store enrollment photos (optional)
        """
        self.output_dir = output_dir
        if output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_enrollment: Optional[Dict] = None
        logger.info("EnrollmentMode initialized")

    def start_enrollment(self, name: str) -> Dict:
        """
        Start a new enrollment session.

        Args:
            name: Person's name for identification

        Returns:
            Enrollment session dict
        """
        self.current_enrollment = {
            "name": name,
            "start_time": datetime.now(),
            "poses": {pose: [] for pose in self.REQUIRED_POSES},
            "completed_poses": set(),
        }
        logger.info(f"Started enrollment for: {name}")
        return self.current_enrollment

    def add_frame(
        self,
        face_image: np.ndarray,
        pose_label: str,
        confidence: float,
        pose_angles: Optional[Dict] = None,
    ) -> Dict:
        """
        Add a captured frame to the current enrollment.

        Args:
            face_image: Aligned RGB face image
            pose_label: Pose classification (front, left, right, etc.)
            confidence: Head pose detection confidence
            pose_angles: Detailed angle measurements

        Returns:
            Status dict with progress information
        """
        if self.current_enrollment is None:
            return {"status": "error", "message": "No active enrollment"}

        if pose_label not in self.REQUIRED_POSES:
            return {
                "status": "error",
                "message": f"Invalid pose: {pose_label}",
            }

        # Check if pose already has enough frames
        if len(self.current_enrollment["poses"][pose_label]) >= self.FRAMES_PER_POSE:
            return {
                "status": "skip",
                "message": f"Already have {self.FRAMES_PER_POSE} frames for {pose_label}",
            }

        # Store frame
        frame_data = {
            "image": face_image.copy(),
            "confidence": confidence,
            "angles": pose_angles or {},
            "timestamp": datetime.now(),
        }
        self.current_enrollment["poses"][pose_label].append(frame_data)
        self.current_enrollment["completed_poses"].add(pose_label)

        frames_count = len(self.current_enrollment["poses"][pose_label])
        remaining = self.FRAMES_PER_POSE - frames_count

        logger.info(
            f"Enrollment frame added: {pose_label} ({frames_count}/{self.FRAMES_PER_POSE})"
        )

        return {
            "status": "captured",
            "pose": pose_label,
            "frames_for_pose": frames_count,
            "remaining": remaining,
            "total_poses_started": len(self.current_enrollment["completed_poses"]),
            "total_poses_required": len(self.REQUIRED_POSES),
        }

    def get_enrollment_progress(self) -> Dict:
        """Get current enrollment progress."""
        if self.current_enrollment is None:
            return {"status": "no_enrollment"}

        progress = {
            "name": self.current_enrollment["name"],
            "total_poses": len(self.REQUIRED_POSES),
            "completed_poses": len(self.current_enrollment["completed_poses"]),
            "pose_status": {},
        }

        for pose in self.REQUIRED_POSES:
            frames = len(self.current_enrollment["poses"][pose])
            progress["pose_status"][pose] = {
                "frames": frames,
                "required": self.FRAMES_PER_POSE,
                "complete": frames >= self.FRAMES_PER_POSE,
            }

        progress["is_complete"] = all(
            prog["complete"]
            for prog in progress["pose_status"].values()
        )

        return progress

    def finish_enrollment(self) -> Dict:
        """
        Finish the current enrollment and return all captured data.

        Returns:
            Complete enrollment data with all frames and metadata
        """
        if self.current_enrollment is None:
            return {"status": "error", "message": "No active enrollment"}

        progress = self.get_enrollment_progress()
        if not progress["is_complete"]:
            return {
                "status": "incomplete",
                "message": "Not all poses have sufficient frames",
                "progress": progress,
            }

        enrollment_data = {
            "status": "complete",
            "name": self.current_enrollment["name"],
            "enrollment_data": self.current_enrollment,
            "timestamp": datetime.now(),
        }

        # Save enrollment photos if output_dir is set
        if self.output_dir is not None:
            self._save_enrollment_photos()

        self.current_enrollment = None
        logger.info(f"Enrollment completed")

        return enrollment_data

    def _save_enrollment_photos(self) -> None:
        """Save all enrollment photos to disk."""
        if self.current_enrollment is None or self.output_dir is None:
            return

        enrollment_dir = (
            self.output_dir
            / self.current_enrollment["name"].replace(" ", "_")
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        enrollment_dir.mkdir(parents=True, exist_ok=True)

        for pose, frames in self.current_enrollment["poses"].items():
            pose_dir = enrollment_dir / pose
            pose_dir.mkdir(exist_ok=True)

            for i, frame_data in enumerate(frames):
                filename = pose_dir / f"frame_{i+1:02d}.jpg"
                cv2.imwrite(str(filename), cv2.cvtColor(frame_data["image"], cv2.COLOR_RGB2BGR))

        logger.info(f"Enrollment photos saved to: {enrollment_dir}")

    def cancel_enrollment(self) -> Dict:
        """Cancel the current enrollment."""
        if self.current_enrollment is None:
            return {"status": "error", "message": "No active enrollment"}

        name = self.current_enrollment["name"]
        self.current_enrollment = None
        logger.info(f"Enrollment cancelled for: {name}")

        return {"status": "cancelled", "name": name}


def draw_enrollment_guide(
    frame: np.ndarray,
    current_pose: str,
    frame_count: int,
    progress: Dict,
) -> np.ndarray:
    """
    Draw enrollment UI guide on frame.

    Args:
        frame: Input frame
        current_pose: Current pose being captured
        frame_count: Number of frames captured for this pose
        progress: Enrollment progress dict

    Returns:
        Frame with drawn UI elements
    """
    h, w = frame.shape[:2]

    # Draw semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Draw title
    cv2.putText(
        frame,
        "ENROLLMENT MODE - Multi-Angle Face Capture",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    # Draw current pose instruction
    pose_descriptions = {
        "front": "🎯 Look STRAIGHT at camera",
        "left_profile": "🎯 Turn head LEFT (90°)",
        "right_profile": "🎯 Turn head RIGHT (90°)",
        "up": "🎯 Look UP (tilt head back)",
        "down": "🎯 Look DOWN (tilt head down)",
    }

    instruction = pose_descriptions.get(current_pose, "Unknown pose")
    cv2.putText(
        frame,
        instruction,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
    )

    # Draw frame counter
    cv2.putText(
        frame,
        f"Frames: {frame_count}/3",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )

    # Draw pose progress
    start_y = 220
    for i, pose in enumerate(EnrollmentMode.REQUIRED_POSES):
        pose_prog = progress["pose_status"][pose]
        status_text = (
            "✓" if pose_prog["complete"] else
            f"({pose_prog['frames']}/{pose_prog['required']})"
        )
        color = (0, 255, 0) if pose_prog["complete"] else (100, 100, 100)

        cv2.putText(
            frame,
            f"{pose}: {status_text}",
            (20, start_y + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Draw completion percentage
    total_frames = sum(
        len(frames)
        for frames in progress.get("enrollment_data", {}).get("poses", {}).values()
    )
    total_required = len(EnrollmentMode.REQUIRED_POSES) * EnrollmentMode.FRAMES_PER_POSE
    percentage = int((total_frames / total_required) * 100) if total_required > 0 else 0

    cv2.putText(
        frame,
        f"Overall Progress: {percentage}%",
        (20, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    # Draw hint text
    cv2.putText(
        frame,
        "Press 'S' to capture | 'N' next pose | 'Q' quit enrollment",
        (20, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )

    return frame
