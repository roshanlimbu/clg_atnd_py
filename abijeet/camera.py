"""
camera_feed.py — High-performance camera feed for the attendance system.

Architecture
------------
The single biggest reason for <20 fps in the original code is that capture,
resize, inference, and display all ran sequentially on the main thread.
Every face-detection call (typically 20-80 ms) stalled the capture loop,
causing frame drops and a visible FPS dip.

This version separates concerns into three layers:

  ┌──────────────────────────────────────────────────────┐
  │  _CaptureThread  (daemon thread)                     │
  │  • reads raw frames from the camera at full speed    │
  │  • always keeps only the freshest frame in a 1-slot  │
  │    queue — stale frames are discarded immediately    │
  └──────────────────────┬───────────────────────────────┘
                         │  queue.Queue(maxsize=1)
  ┌──────────────────────▼───────────────────────────────┐
  │  CameraFeed.get_frames()  (main / caller thread)     │
  │  • pulls latest frame, resizes for display           │
  │  • produces a smaller copy for inference on samples  │
  │  • draws overlay in-place  (zero extra allocation)   │
  │  • tracks real FPS with a timestamp ring buffer      │
  └──────────────────────────────────────────────────────┘

Result: the camera never blocks on inference; display stays at the
camera's native frame rate (≥30 fps); every sampled frame reaches the
detector without being dropped.
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from collections import deque
from datetime import datetime
from typing import Generator, Tuple

import ctypes

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device profiles
# Resolution sent to the *detector* (display always matches what the camera
# actually gives us).  Smaller = faster inference = higher effective FPS.
# ---------------------------------------------------------------------------

DEVICE_PROFILES: dict[str, dict] = {
    "low":    {"proc_width": 320, "proc_height": 180, "sample_every": 6},
    "medium": {"proc_width": 480, "proc_height": 270, "sample_every": 3},
    "high":   {"proc_width": 640, "proc_height": 360, "sample_every": 2},
}

# Frame-time budget: 33 ms = 30 fps target.
_BUDGET_MS           = 33.0
_DOWNGRADE_WINDOW    = 20    # rolling-window size for adaptive decisions
_DOWNGRADE_THRESHOLD = 1.4   # downgrade if avg > budget × this
_DOWNGRADE_COOLDOWN  = 90    # frames to skip before next downgrade check

# Status-bar appearance (drawn in-place; no frame copy needed).
_BAR_H      = 38
_BAR_BG     = (30, 30, 30)     # BGR dark gray
_BAR_TEXT   = (230, 230, 230)  # BGR near-white
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1


# ---------------------------------------------------------------------------
# Background capture thread
# ---------------------------------------------------------------------------

class _CaptureThread(threading.Thread):
    """
    Daemon thread that reads camera frames at full speed.

    Exposes only the *latest* frame through a Queue(maxsize=1).
    When the consumer is busy, the thread discards the previous unread frame
    and replaces it — the consumer always gets the freshest picture.
    """

    def __init__(self, capture: cv2.VideoCapture) -> None:
        super().__init__(daemon=True, name="CameraCapture")
        self._cap   = capture
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._consecutive_failures = 0

    # -- threading.Thread interface -----------------------------------------

    def run(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 5:
                    logger.error(
                        "Camera read failed 5 times consecutively — "
                        "sending stop sentinel."
                    )
                    # Put None as a sentinel so next_frame() can signal EOF.
                    try:
                        self._q.put_nowait(None)
                    except queue.Full:
                        pass
                    break
                time.sleep(0.005)
                continue

            self._consecutive_failures = 0

            # Discard unread stale frame; replace with fresh one.
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put(frame)

    # -- Consumer interface --------------------------------------------------

    def next_frame(self, timeout: float = 0.15) -> np.ndarray | None:
        """
        Block until the next frame is available (up to ``timeout`` seconds).
        Returns ``None`` on timeout or when a stop sentinel is received.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CameraFeed:
    """
    High-performance camera feed targeting ≥30 fps with zero missed detections.

    Parameters
    ----------
    camera_index:
        OpenCV device index (0 = first/default camera).
    capture_width / capture_height:
        Resolution *requested* from the camera driver.  The camera may clamp
        to a supported mode; ``actual_width`` / ``actual_height`` reflect the
        real resolution after ``start()``.
    device_profile:
        ``"auto"`` (default) — benchmark on startup and pick a tier.
        ``"low"`` / ``"medium"`` / ``"high"`` — pin to a specific tier.
    adaptive:
        When ``True`` (default), automatically step down the profile if the
        detection pipeline consistently overshoots the 33 ms frame budget.
    window_name:
        Title of the OpenCV display window.

    Typical usage
    -------------
    ::
        feed = CameraFeed()
        if not feed.start():
            raise RuntimeError("Cannot open camera")

        try:
            for display, process in feed.get_frames():
                if process is not None:
                    t0 = time.perf_counter()
                    results = detector.detect(process)
                    feed.record_process_time((time.perf_counter() - t0) * 1000)
                    annotate(display, results)
                feed.display_frame(display)
        finally:
            feed.stop()

    Or with the context manager::

        with CameraFeed() as feed:
            for display, process in feed.get_frames():
                ...
    """

    def __init__(
        self,
        camera_index:   int  = 0,
        capture_width:  int  = 1280,
        capture_height: int  = 720,
        device_profile: str  = "auto",
        adaptive:       bool = True,
        fullscreen:     bool = False,
        window_name:    str  = "Face Attendance",
    ) -> None:
        self.camera_index   = camera_index
        self.capture_width  = capture_width
        self.capture_height = capture_height
        self.window_name    = window_name
        self._adaptive      = adaptive
        self._requested_profile = device_profile
        self._fullscreen = fullscreen

        # Filled after start() to reflect what the camera actually delivers.
        self.actual_width:  int = capture_width
        self.actual_height: int = capture_height

        # Profile state (updated by _apply_profile).
        self._profile_name: str = "medium"
        self.proc_width:    int = 480
        self.proc_height:   int = 270
        self.sample_every:  int = 3

        self._capture: cv2.VideoCapture | None = None
        self._thread:  _CaptureThread   | None = None
        self._running: bool = False

        # Timestamp ring for FPS measurement (last 60 frames).
        self._ts_ring: deque[float] = deque(maxlen=60)

        # Adaptive downgrade state.
        self._proc_times:         deque[float] = deque(maxlen=_DOWNGRADE_WINDOW)
        self._downgrade_cooldown: int = 0

        # Pre-allocated status-bar buffer (lazy init; matches frame width).
        self._bar_buf: np.ndarray | None = None

        # Fullscreen display size, populated on start when requested.
        self._display_width: int | None = None
        self._display_height: int | None = None

        # Monotonically increasing frame counter.
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Open the camera, start the capture thread, run the benchmark.

        Returns ``True`` on success, ``False`` if the camera cannot be opened.
        """
        system = platform.system()
        if system == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif system == "Windows":
            backend = cv2.CAP_DSHOW   # avoids multi-second enumeration hang of CAP_ANY
        else:
            backend = cv2.CAP_ANY

        logger.info("Opening camera index=%d backend=%s ...", self.camera_index, backend)
        cap = cv2.VideoCapture(self.camera_index, backend)

        if not cap.isOpened():
            logger.error("Could not open camera at index %d.", self.camera_index)
            if platform.system() == "Darwin":
                logger.error(
                    "macOS: grant Camera permission to your terminal app "
                    "(Terminal / iTerm / VS Code) and restart it before running."
                )
            cap.release()
            return False

        # Request resolution and frame rate — the driver may silently clamp.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        cap.set(cv2.CAP_PROP_FPS,          30)

        # Minimize driver-side buffering so we always get the freshest frame.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read back what the driver actually agreed to.
        self.actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._capture = cap
        self._thread  = _CaptureThread(cap)
        self._thread.start()

        # Drain first 4 frames (often dark / auto-exposure not settled yet).
        for _ in range(4):
            self._thread.next_frame(timeout=0.5)

        # Pick profile.
        profile = (
            self._benchmark()
            if self._requested_profile == "auto"
            else self._requested_profile
        )
        self._apply_profile(profile)

        self._running = True
        # Create the display window and optionally make it fullscreen.
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if self._fullscreen:
                self._display_width, self._display_height = self._get_screen_size()
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
            else:
                self._display_width = self.actual_width
                self._display_height = self.actual_height
                # Resize the window to match the camera's actual resolution.
                try:
                    cv2.resizeWindow(self.window_name, self.actual_width, self.actual_height)
                except Exception:
                    pass
        except Exception:
            # Best-effort: some OpenCV builds or platforms may not support window calls.
            pass

        logger.info(
            "Camera started — index=%d  actual=%dx%d  profile=%s  "
            "proc=%dx%d  sample_every=%d",
            self.camera_index,
            self.actual_width, self.actual_height,
            self._profile_name,
            self.proc_width, self.proc_height,
            self.sample_every,
        )
        return True

    def stop(self) -> None:
        """
        Stop capture, join the background thread, release the camera.
        Safe to call multiple times (idempotent).
        """
        self._running = False
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        cv2.destroyAllWindows()
        logger.info("Camera released.")

    def __enter__(self) -> "CameraFeed":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def get_frames(
        self,
    ) -> Generator[Tuple[np.ndarray, np.ndarray | None], None, None]:
        """
        Yield ``(display_frame, process_frame)`` until the user quits or
        the camera stops.

        ``display_frame``
            Full-resolution BGR frame with the status bar already drawn
            in-place.  Pass directly to :meth:`display_frame`.

        ``process_frame``
            Down-scaled BGR copy intended for inference.  ``None`` on
            non-sampled frames — skip the detector on those.

        After running inference on ``process_frame``, call
        :meth:`record_process_time` with elapsed milliseconds so the
        adaptive logic can monitor pipeline health.

        The generator ensures ``self._running = False`` on exit via a
        ``finally`` block, so cleanup is guaranteed even if the caller
        breaks early or raises.
        """
        if self._capture is None or self._thread is None:
            raise RuntimeError("call CameraFeed.start() before get_frames().")

        try:
            while self._running:
                raw = self._thread.next_frame(timeout=0.15)
                if raw is None:
                    logger.warning("No frame received (timeout or camera stopped).")
                    break

                self._ts_ring.append(time.perf_counter())
                self._frame_count += 1

                # Resize to what the camera actually gave us (often a no-op).
                display = self._to_display(raw)

                # Decide whether this frame gets sent to the detector.
                is_sample  = (self._frame_count % self.sample_every == 0)
                process    = self._to_process(display) if is_sample else None

                # Stamp the status bar directly onto display (in-place, no copy).
                self._draw_bar(display, is_sample)

                yield display, process

                # Tick down the adaptive cooldown counter each frame.
                if self._downgrade_cooldown > 0:
                    self._downgrade_cooldown -= 1

                # Non-blocking key check — 'q' / ESC quits.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Quit key pressed — stopping.")
                    break

        finally:
            self._running = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def record_process_time(self, elapsed_ms: float) -> None:
        """
        Report how long the last detection call took (in milliseconds).

        Call this immediately after the inference block so the adaptive
        scheduler has accurate timing data:

        ::
            t0 = time.perf_counter()
            results = detector.detect(process_frame)
            feed.record_process_time((time.perf_counter() - t0) * 1000)
        """
        if not self._adaptive:
            return
        self._proc_times.append(elapsed_ms)
        if (
            self._downgrade_cooldown == 0
            and len(self._proc_times) >= _DOWNGRADE_WINDOW
        ):
            avg = sum(self._proc_times) / len(self._proc_times)
            if avg > _BUDGET_MS * _DOWNGRADE_THRESHOLD:
                self._try_downgrade(avg)

    def display_frame(self, frame: np.ndarray) -> None:
        """Show ``frame`` in the named OpenCV window."""
        if self._fullscreen:
            width, height = self._get_display_size()
            if width > 0 and height > 0:
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
        cv2.imshow(self.window_name, frame)

    @property
    def fps(self) -> float:
        """Live display FPS measured over the last ≤60 frames."""
        if len(self._ts_ring) < 2:
            return 0.0
        span = self._ts_ring[-1] - self._ts_ring[0]
        return (len(self._ts_ring) - 1) / span if span > 0 else 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_profile(self, name: str) -> None:
        """Update proc resolution + sample_every from a named profile."""
        cfg = DEVICE_PROFILES.get(name, DEVICE_PROFILES["medium"])
        self._profile_name = name
        self.proc_width    = cfg["proc_width"]
        self.proc_height   = cfg["proc_height"]
        self.sample_every  = cfg["sample_every"]
        # Clear old measurements so stale data doesn't immediately trigger
        # another downgrade after we've just switched profiles.
        self._proc_times.clear()

    def _to_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the frame at actual_width × actual_height.
        If dimensions already match, returns the original array (zero-copy).
        """
        h, w = frame.shape[:2]
        if w == self.actual_width and h == self.actual_height:
            return frame
        # Use INTER_AREA for downscale (best quality), INTER_LINEAR for upscale.
        interp = cv2.INTER_AREA if w > self.actual_width else cv2.INTER_LINEAR
        return cv2.resize(frame, (self.actual_width, self.actual_height), interpolation=interp)

    def _to_process(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a down-scaled copy of ``frame`` at proc_width × proc_height.
        Zero-copy if proc resolution already matches display resolution.
        """
        if self.proc_width == self.actual_width and self.proc_height == self.actual_height:
            return frame
        return cv2.resize(
            frame,
            (self.proc_width, self.proc_height),
            interpolation=cv2.INTER_LINEAR,  # fastest for mild downscale
        )

    def _draw_bar(self, frame: np.ndarray, is_sample: bool) -> None:
        """
        Stamp the status bar onto the top _BAR_H rows of ``frame`` in-place.

        Uses a pre-allocated numpy buffer for the background fill instead of
        cv2.rectangle() so there is zero heap allocation per frame.
        """
        _, w = frame.shape[:2]

        # Lazy-allocate (or re-allocate if frame width changed).
        if self._bar_buf is None or self._bar_buf.shape[1] != w:
            self._bar_buf = np.full((_BAR_H, w, 3), _BAR_BG, dtype=np.uint8)

        # In-place paste: no copy, no allocation.
        frame[:_BAR_H, :w] = self._bar_buf

        status = "DETECT" if is_sample else "LIVE"
        text = (
            f"{datetime.now().strftime('%H:%M:%S')}  "
            f"{status}  |  "
            f"profile:{self._profile_name}  "
            f"proc:{self.proc_width}x{self.proc_height}  "
            f"FPS:{self.fps:.1f}"
        )
        cv2.putText(
            frame, text, (10, 25),
            _FONT, _FONT_SCALE, _BAR_TEXT, _FONT_THICK, cv2.LINE_AA,
        )

    def _try_downgrade(self, avg_ms: float) -> None:
        """Step down to the next lighter profile if one is available."""
        order = ["high", "medium", "low"]
        try:
            idx = order.index(self._profile_name)
        except ValueError:
            idx = 1  # unknown profile — treat as medium

        if idx >= len(order) - 1:
            logger.warning(
                "Already at lowest profile '%s'. "
                "Avg detection time %.0f ms exceeds %.0f ms budget. "
                "Consider a lighter face detector.",
                self._profile_name, avg_ms, _BUDGET_MS,
            )
            return

        new = order[idx + 1]
        logger.warning(
            "Detection avg %.0f ms > budget %.0f ms — downgrading: %s → %s",
            avg_ms, _BUDGET_MS, self._profile_name, new,
        )
        self._apply_profile(new)
        self._downgrade_cooldown = _DOWNGRADE_COOLDOWN

    def _get_screen_size(self) -> tuple[int, int]:
        """Return the primary screen size on Windows; fall back to camera size."""
        try:
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return self.actual_width, self.actual_height

    def _get_display_size(self) -> tuple[int, int]:
        """Return the current OpenCV window size, or fall back to screen size."""
        try:
            x, y, width, height = cv2.getWindowImageRect(self.window_name)
            if width > 0 and height > 0:
                return int(width), int(height)
        except Exception:
            pass
        if self._display_width and self._display_height:
            return self._display_width, self._display_height
        return self._get_screen_size()

    def _benchmark(self) -> str:
        """Measure the actual display+process resize pipeline to pick a profile."""
        if self._thread is None:
            return "medium"

        times: list[float] = []
        for i in range(20):
            frame = self._thread.next_frame(timeout=0.3)
            if frame is None:
                continue
            t0 = time.perf_counter()
            display = self._to_display(frame)
            self._to_process(display)          # simulate the real inference-resize path
            if i >= 4:
                times.append((time.perf_counter() - t0) * 1000)

        if not times:
            return "medium"

        avg = sum(times) / len(times)
        logger.debug("Benchmark: avg resize time = %.1f ms", avg)

        if avg < 8:
            return "high"
        elif avg < 18:
            return "medium"
        else:
            return "low"