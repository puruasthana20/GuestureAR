"""
modules/utils.py
=================
Generic helper functions shared across the project: logging setup,
screenshot saving, drawing helpers, and bounding-box math.

Keeping these here (instead of duplicating logic inside camera.py /
hand_tracker.py / app.py) follows the DRY (Don't Repeat Yourself)
principle demanded by an industrial-quality codebase.
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configures a project-wide logger that writes to BOTH:
      1. Console  - for live debugging while running app.py
      2. A timestamped file inside /logs - for postmortem debugging after
         the program has closed (e.g. crash reports, camera failures).

    Args:
        log_dir: directory where the log file will be created.
        log_level: one of DEBUG / INFO / WARNING / ERROR / CRITICAL.

    Returns:
        A configured `logging.Logger` instance. Every other module calls
        `logging.getLogger("GestureAR")` to reuse this exact configuration
        without having to re-run setup.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{timestamp}.log"

    logger = logging.getLogger("GestureAR")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Guard against duplicate handlers if setup_logging() is accidentally
    # called more than once in the same process (e.g. during testing).
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Log file: %s", log_file)
    return logger


def save_screenshot(frame: np.ndarray, screenshot_dir: Path, logger: logging.Logger) -> str:
    """
    Saves the current (annotated) frame as a timestamped PNG file.

    Args:
        frame: the BGR image (numpy array) to save.
        screenshot_dir: destination folder.
        logger: logger instance used to record success/failure.

    Returns:
        The saved file path as a string (useful for tests/automation).
    """
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = screenshot_dir / f"capture_{timestamp}.png"
    success = cv2.imwrite(str(file_path), frame)
    if success:
        logger.info("Screenshot saved: %s", file_path)
    else:
        logger.error("Failed to save screenshot to %s", file_path)
    return str(file_path)


def calculate_bounding_box(
    landmark_points: List[Tuple[int, int]], padding: int = 20
) -> Tuple[int, int, int, int]:
    """
    Computes an axis-aligned bounding box around a set of (x, y) pixel
    points (typically the 21 hand landmarks).

    Args:
        landmark_points: list of (x, y) pixel coordinates.
        padding: extra pixels added on every side so the box doesn't hug
                 the fingertips too tightly. Also gives Phase 2 gesture
                 modules (which may crop this region as an ROI) some
                 margin for finger movement between frames.

    Returns:
        (x_min, y_min, x_max, y_max). Returns (0, 0, 0, 0) if the input
        list is empty (i.e. no hand detected).
    """
    if not landmark_points:
        return (0, 0, 0, 0)

    xs = [p[0] for p in landmark_points]
    ys = [p[1] for p in landmark_points]

    x_min, x_max = min(xs) - padding, max(xs) + padding
    y_min, y_max = min(ys) - padding, max(ys) + padding

    return (x_min, y_min, x_max, y_max)


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    """
    Draws `text` with a filled rectangle behind it so it stays readable
    against any webcam background (unpredictable lighting/colors).
    Modifies `frame` IN PLACE (no return value) to avoid extra copies
    on every call, which matters for real-time performance.

    Args:
        frame: image to draw on.
        text: string to render.
        position: (x, y) = bottom-left corner of the text baseline.
        font_scale, color, bg_color, thickness: standard OpenCV text params.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    cv2.rectangle(
        frame,
        (x - 4, y - text_h - 8),
        (x + text_w + 4, y + baseline),
        bg_color,
        thickness=-1,  # -1 = filled rectangle
    )
    cv2.putText(frame, text, (x, y - 4), font, font_scale, color, thickness, cv2.LINE_AA)


def clamp(value: int, min_value: int, max_value: int) -> int:
    """
    Clamps `value` into the inclusive range [min_value, max_value].
    Used to keep bounding boxes / landmark pixel coordinates from going
    outside the visible frame (which would otherwise crash cv2.rectangle
    or draw off-screen).
    """
    return max(min_value, min(value, max_value))
