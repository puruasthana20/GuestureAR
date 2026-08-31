"""
config.py
==========
Centralized configuration for the AR-Based Gesture Controlled System.

Design rationale:
- All tunable parameters live in ONE place so that every module (camera,
  hand tracker, app loop) reads consistent values instead of hardcoding
  magic numbers across the codebase. This is critical for an "industrial
  quality" project: when Phase 2 adds gesture classification or a PyQt5
  GUI, those modules will import the SAME config object and never drift
  out of sync with camera/tracker settings.
- Python's `dataclass` gives us type hints, default values, and easy
  extensibility without boilerplate __init__ methods.
- A single module-level `config` instance is created at the bottom of
  this file and imported everywhere else (`from config import config`).
  This is the standard "singleton via module" pattern in Python - since
  modules are only ever imported once and cached in sys.modules, every
  file that imports `config` gets the exact same object in memory.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraConfig:
    """Settings related to webcam capture."""
    device_index: int = 0          # 0 = default system webcam. Change to 1, 2... for external cams.
    frame_width: int = 1280        # Requested capture width in pixels
    frame_height: int = 720        # Requested capture height in pixels
    target_fps: int = 30           # Requested capture frame rate (hardware dependent)
    flip_horizontal: bool = True   # Mirror the feed -> matches user's intuitive left/right movement
    buffer_size: int = 1           # Keep OpenCV's internal frame buffer small -> lower latency


@dataclass
class HandTrackerConfig:
    """Settings for the MediaPipe Hands solution."""
    max_num_hands: int = 2                     # Detect up to 2 hands simultaneously
    model_complexity: int = 1                   # 0 = lite (faster, less accurate), 1 = full (default)
    min_detection_confidence: float = 0.7       # Threshold to accept a NEW hand detection
    min_tracking_confidence: float = 0.6        # Threshold to keep tracking an existing hand
    static_image_mode: bool = False             # False = video mode (uses tracking between frames)


@dataclass
class DisplayConfig:
    """Settings for the on-screen overlay (all toggleable at runtime via hotkeys)."""
    show_fps: bool = True
    show_landmark_ids: bool = False              # Verbose; toggle at runtime with the 'l' key
    show_bounding_box: bool = True
    show_handedness_label: bool = True
    window_name: str = "AR Gesture System - Phase 1 (Hand Tracking)"


@dataclass
class PathConfig:
    """Filesystem locations used by the app. Auto-creates required folders on startup."""
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    log_dir: Path = field(init=False)
    screenshot_dir: Path = field(init=False)

    def __post_init__(self):
        self.log_dir = self.root_dir / "logs"
        self.screenshot_dir = self.root_dir / "screenshots"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Top-level config object that aggregates every sub-config into one place."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    log_level: str = "INFO"   # DEBUG | INFO | WARNING | ERROR | CRITICAL


# Single shared instance imported by every module -> "single source of truth".
# Example usage elsewhere: `from config import config; config.camera.frame_width`
config = AppConfig()
