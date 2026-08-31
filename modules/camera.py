"""
modules/camera.py
===================
Wraps cv2.VideoCapture with:
  - robust initialization + error handling (raises a clear, catchable
    exception instead of silently returning black frames)
  - configurable resolution / target FPS / buffer size
  - context-manager support (`with Camera(...) as cam:`) for guaranteed
    resource cleanup even if an exception occurs mid-loop
  - structured logging of every lifecycle event

Isolating camera logic here means app.py - and any future module such as
a PyQt5 GUI in Phase 2 - can reuse this exact, tested camera interface
instead of re-implementing raw cv2.VideoCapture calls in multiple places.
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple

from config import CameraConfig


class CameraError(Exception):
    """Raised when the webcam cannot be opened or fails during operation."""
    pass


class Camera:
    """
    High-level webcam interface built on top of cv2.VideoCapture.

    Usage:
        with Camera(config.camera, logger) as cam:
            ret, frame = cam.read()
            if ret:
                ... process frame ...
        # camera is automatically released here, even on exception
    """

    def __init__(self, cam_config: CameraConfig, logger: logging.Logger):
        self.config = cam_config
        self.logger = logger
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """
        Opens the webcam device and applies resolution/FPS/buffer settings.

        Raises:
            CameraError: if the device cannot be opened by any backend.
        """
        self.logger.info("Opening camera device index=%s", self.config.device_index)

        # CAP_DSHOW improves startup reliability/speed on Windows.
        # On Linux/Mac this backend hint is simply ignored by OpenCV,
        # so it is always safe to try first.
        self.cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.logger.warning("CAP_DSHOW open failed, retrying with default backend.")
            self.cap = cv2.VideoCapture(self.config.device_index)

        if not self.cap.isOpened():
            self.logger.error("Failed to open camera at index %s", self.config.device_index)
            raise CameraError(
                f"Cannot open webcam at index {self.config.device_index}. "
                f"Check that it is physically connected, that no other "
                f"application (Zoom, Teams, another script) is using it, "
                f"and that OS camera permissions are granted."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(
            "Camera opened successfully. Requested=%dx%d Actual=%dx%d",
            self.config.frame_width, self.config.frame_height, actual_w, actual_h,
        )
        # NOTE: Many webcams silently ignore CAP_PROP_* requests they don't
        # support and fall back to their nearest native resolution - this
        # is why we log BOTH the requested and actual values.

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single frame from the webcam.

        Returns:
            (success, frame): `success` is False and `frame` is None if
            the read failed (e.g. camera unplugged mid-session).
            Applies a horizontal flip if configured, mirroring the feed so
            the user's hand movement matches intuitive left/right motion
            on screen (like a mirror, not a security camera).
        """
        if self.cap is None or not self.cap.isOpened():
            raise CameraError("Camera.read() called before the camera was opened.")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.logger.warning("Failed to read frame from camera.")
            return False, None

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Releases the camera resource. Safe to call multiple times / even if never opened."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released.")
        self.cap = None

    # --- Context manager protocol: enables `with Camera(...) as cam:` ---
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        # Returning False re-raises any exception that occurred inside the
        # `with` block instead of swallowing it - errors must stay visible.
        return False
