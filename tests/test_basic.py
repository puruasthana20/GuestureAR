"""
tests/test_basic.py
=====================
Basic unit tests for Phase 1.

Run with:  pytest -v

Design note:
These tests deliberately avoid requiring a physical webcam or a display
(so they run cleanly in CI / headless environments). They cover the pure
-logic functions: bounding-box math, coordinate clamping, FPS smoothing,
and config sanity. Camera + HandTracker integration (which NEED real
hardware and a visible window) are instead covered by the manual
"Testing Checklist" in README.md.
"""

import sys
import os
import time

# Ensure the project root is on sys.path so `import config` / `import modules...`
# works when pytest is run from the tests/ folder or from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.utils import calculate_bounding_box, clamp
from modules.fps import FPS
from config import AppConfig, CameraConfig, HandTrackerConfig


def test_calculate_bounding_box_basic():
    """Bounding box should tightly enclose all given points when padding=0."""
    points = [(10, 10), (50, 60), (30, 5)]
    x_min, y_min, x_max, y_max = calculate_bounding_box(points, padding=0)
    assert x_min == 10
    assert y_min == 5
    assert x_max == 50
    assert y_max == 60


def test_calculate_bounding_box_with_padding():
    """Padding should expand the box symmetrically on every side."""
    points = [(100, 100)]
    x_min, y_min, x_max, y_max = calculate_bounding_box(points, padding=10)
    assert x_min == 90
    assert y_min == 90
    assert x_max == 110
    assert y_max == 110


def test_calculate_bounding_box_empty():
    """No landmarks (no hand detected) should return a safe zero-box, not crash."""
    assert calculate_bounding_box([]) == (0, 0, 0, 0)


def test_clamp_within_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_min():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above_max():
    assert clamp(15, 0, 10) == 10


def test_fps_returns_positive_value():
    """After two update() calls with real elapsed time, FPS should be > 0."""
    fps_meter = FPS()
    fps_meter.update()
    time.sleep(0.05)
    value = fps_meter.update()
    assert value > 0


def test_fps_get_matches_last_update():
    fps_meter = FPS()
    fps_meter.update()
    time.sleep(0.02)
    updated_value = fps_meter.update()
    assert fps_meter.get() == updated_value


def test_camera_config_defaults():
    cam_config = CameraConfig()
    assert cam_config.frame_width == 1280
    assert cam_config.frame_height == 720
    assert cam_config.device_index == 0


def test_hand_tracker_config_defaults():
    tracker_config = HandTrackerConfig()
    assert 0.0 <= tracker_config.min_detection_confidence <= 1.0
    assert 0.0 <= tracker_config.min_tracking_confidence <= 1.0
    assert tracker_config.max_num_hands >= 1


def test_app_config_creates_directories():
    """AppConfig() should auto-create logs/ and screenshots/ on instantiation."""
    app_config = AppConfig()
    assert app_config.paths.log_dir.exists()
    assert app_config.paths.screenshot_dir.exists()
