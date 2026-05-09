"""
Frame buffer for selecting the sharpest image from recent frames.

When a person is recognized, this buffer allows selecting the highest-quality
(sharpest) image from the last N frames, rather than just the first detection.
This significantly improves photo evidence quality.
"""

import logging
from collections import defaultdict
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SharpFrameBuffer:
    """
    Buffers recent face crops and provides the sharpest one for a given person.
    """
    
    # Keep last 5 frames per person (at 30fps, ~0.17 seconds of data)
    BUFFER_SIZE = 5
    
    def __init__(self, buffer_size: int = BUFFER_SIZE):
        """Initialize the frame buffer."""
        self.buffer_size = buffer_size
        self.buffers = defaultdict(list)  # person_id -> list of (image, sharpness_score)
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Higher values = sharper image
        Lower values = blurrier image
        
        Args:
            image: BGR or RGB face crop
            
        Returns:
            Sharpness score (0.0+)
        """
        if image is None or image.size == 0:
            return 0.0
        
        try:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(sharpness)
        except Exception as e:
            logger.debug(f"Sharpness calculation error: {e}")
            return 0.0
    
    def add_frame(self, person_id: str, image: np.ndarray) -> None:
        """
        Add a face crop to the buffer for this person.
        
        Args:
            person_id: The person's ID
            image: Face crop image (BGR or RGB)
        """
        if image is None or image.size == 0:
            return
        
        sharpness = self.calculate_sharpness(image)
        
        # Add to buffer
        self.buffers[person_id].append((image.copy(), sharpness))
        
        # Keep only the most recent N frames
        if len(self.buffers[person_id]) > self.buffer_size:
            self.buffers[person_id].pop(0)
        
        logger.debug(
            f"Buffered frame for {person_id}: sharpness={sharpness:.1f} "
            f"(buffer size: {len(self.buffers[person_id])})"
        )
    
    def get_sharpest_frame(self, person_id: str) -> Optional[np.ndarray]:
        """
        Get the sharpest frame from the buffer for this person.
        
        Args:
            person_id: The person's ID
            
        Returns:
            The sharpest image, or None if buffer is empty
        """
        if person_id not in self.buffers or not self.buffers[person_id]:
            return None
        
        # Find the sharpest frame
        sharpest_image, max_sharpness = max(
            self.buffers[person_id],
            key=lambda x: x[1]
        )
        
        logger.debug(
            f"Selected sharpest frame for {person_id}: "
            f"sharpness={max_sharpness:.1f} from {len(self.buffers[person_id])} frames"
        )
        
        return sharpest_image
    
    def get_sharpest_frame_with_score(self, person_id: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the sharpest frame and its sharpness score.
        
        Args:
            person_id: The person's ID
            
        Returns:
            (image, sharpness_score) or (None, 0.0) if buffer is empty
        """
        if person_id not in self.buffers or not self.buffers[person_id]:
            return None, 0.0
        
        return max(self.buffers[person_id], key=lambda x: x[1])
    
    def clear_buffer(self, person_id: str) -> None:
        """Clear the buffer for a specific person."""
        if person_id in self.buffers:
            del self.buffers[person_id]
            logger.debug(f"Cleared buffer for {person_id}")
    
    def clear_all(self) -> None:
        """Clear all buffers."""
        self.buffers.clear()
        logger.debug("Cleared all frame buffers")
    
    def get_stats(self) -> dict:
        """Get statistics about current buffers."""
        return {
            "total_persons": len(self.buffers),
            "total_frames": sum(len(frames) for frames in self.buffers.values()),
            "persons": {
                person_id: len(frames)
                for person_id, frames in self.buffers.items()
            }
        }
