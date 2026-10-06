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
class GestureConfig:
    """
    Settings for the Phase 2 Gesture Recognition Engine (modules/gesture_recognition.py).
    Added alongside the Phase 1 configs above without changing any of them -
    every other dataclass in this file is untouched from Phase 1.
    """
    # --- Finger-state thresholds (see gesture_recognition.py's module docstring
    #     section 1, "Finger Angles", for the geometry behind these numbers) ---
    finger_angle_threshold: float = 160.0   # degrees; above this, a finger counts as EXTENDED
    thumb_angle_threshold: float = 150.0    # thumb's IP-joint angle rarely reaches 160 even when straight
    thumb_tuck_ratio: float = 0.6           # thumb_tip-to-index_MCP distance / palm_size; above this, thumb is "out"

    # --- Distance-ratio thresholds (see section 2, "Distances") ---
    pinch_distance_ratio: float = 0.4       # thumb_tip-to-index_tip distance / palm_size; below this = PINCH
    spread_distance_ratio: float = 0.5      # index_tip-to-middle_tip distance / palm_size; above this = spread (Victory)

    # --- Temporal smoothing (see section 4, "Temporal Tracking") ---
    smoothing_window: int = 7               # number of recent frames considered for the majority vote
    smoothing_majority_ratio: float = 0.6   # fraction of the window that must agree to confirm a gesture
    min_confidence_to_display: float = 0.5  # below this, report Gesture.DETECTING instead of a low-confidence guess

    # --- Swipe (motion) detection ---
    swipe_history_len: int = 10             # frames of wrist position kept for swipe displacement checks
    swipe_min_distance_ratio: float = 1.2   # net displacement / palm_size required to count as a deliberate swipe
    swipe_cooldown_frames: int = 15         # frames to ignore new swipes for, right after one fires

    # --- Display ---
    show_gesture_debug: bool = True         # toggled at runtime with the 'g' key (see app.py)


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
    gesture: GestureConfig = field(default_factory=GestureConfig)  # Phase 2 addition
    paths: PathConfig = field(default_factory=PathConfig)
    log_level: str = "INFO"   # DEBUG | INFO | WARNING | ERROR | CRITICAL


# Single shared instance imported by every module -> "single source of truth".
# Example usage elsewhere: `from config import config; config.camera.frame_width`
config = AppConfig()
