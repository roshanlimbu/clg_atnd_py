"""
Camera feed handling for the attendance system.
"""

import logging
import platform
from datetime import datetime
from typing import Iterator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraFeed:
    """Small wrapper around OpenCV camera capture and display."""

    def __init__(
        self,
        camera_index: int = 0,
        sample_every: int = 5,
        display_width: int = 1280,
        display_height: int = 720,
        window_name: str = "Face Attendance",
    ):
        self.camera_index = camera_index
        self.sample_every = max(1, int(sample_every))
        self.display_width = display_width
        self.display_height = display_height
        self.window_name = window_name

        self._capture = None
        self._running = False
        self._frame_count = 0

    def start(self) -> bool:
        """Open the configured camera."""
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(self.camera_index, backend)
        if not self._capture.isOpened():
            logger.error("Could not open camera index %s", self.camera_index)
            if platform.system() == "Darwin":
                logger.error(
                    "macOS may be blocking camera access. Grant Camera permission "
                    "to the app running Python, such as Terminal, iTerm, or VS Code, "
                    "then restart that app and run again."
                )
            self._capture.release()
            self._capture = None
            return False

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        self._running = True
        logger.info("Camera opened at index %s", self.camera_index)
        return True

    def get_frames(self) -> Iterator[Tuple[np.ndarray, bool]]:
        """Yield `(frame, should_process)` until the camera stops or user quits."""
        if self._capture is None:
            raise RuntimeError("CameraFeed.start() must be called before get_frames()")

        while self._running:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                logger.warning("Failed to read frame from camera")
                break

            frame = self._resize_for_display(frame)
            self._frame_count += 1
            should_process = self._frame_count % self.sample_every == 0
            yield frame, should_process

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Quit key pressed")
                break

        self._running = False

    def display_frame(self, frame: np.ndarray):
        """Show one frame in the OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def draw_status_bar(
        self,
        frame: np.ndarray,
        marked_count: int,
        fps: float,
        processing_frame: bool,
    ) -> np.ndarray:
        """Draw the top status bar with runtime information."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        bar_height = 42

        cv2.rectangle(annotated, (0, 0), (width, bar_height), (32, 32, 32), cv2.FILLED)

        status = "PROCESSING" if processing_frame else "LIVE"
        text = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{status} | Attendance count: {marked_count} | FPS: {fps:.1f}"
        )
        cv2.putText(
            annotated,
            text,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def stop(self):
        """Release camera and close OpenCV windows."""
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        cv2.destroyAllWindows()
        logger.info("Camera released")

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        if self.display_width <= 0 or self.display_height <= 0:
            return frame

        return cv2.resize(
            frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
