"""
modules/hand_tracker.py
=========================
Wraps Google's MediaPipe Hands solution to provide:
  - multi-hand detection (Left / Right / multiple hands simultaneously)
  - 21-landmark extraction per hand, in BOTH pixel and normalized coords
  - a padded bounding box per hand
  - skeleton + label + landmark-ID drawing on the output frame

This module is the "brain" of Phase 1. Every other module (camera, fps,
utils) exists to feed frames into this module and display its output.
Phase 2 (gesture classification) will consume the `HandData` objects this
module returns without needing to touch MediaPipe directly at all.
"""

import cv2
import logging
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from config import HandTrackerConfig, DisplayConfig
from modules.utils import calculate_bounding_box, draw_text_with_background, clamp


@dataclass
class HandData:
    """
    Structured result for a single detected hand.

    Using a dataclass (instead of raw dicts/tuples) makes downstream
    Phase 2 gesture-classification code type-safe, IDE-autocompletable,
    and self-documenting.
    """
    label: str                                          # "Left" or "Right"
    confidence: float                                    # handedness classification confidence (0-1)
    landmarks_px: List[Tuple[int, int]]                  # 21 x (x, y) pixel coordinates
    landmarks_norm: List[Tuple[float, float, float]]     # 21 x (x, y, z) normalized [0-1] coordinates
    bounding_box: Tuple[int, int, int, int]              # (x_min, y_min, x_max, y_max) in pixels


class HandTracker:
    """
    High-level interface around mediapipe.solutions.hands.

    Typical per-frame usage:
        hands_data, annotated_frame = tracker.process(frame)
        for hand in hands_data:
            print(hand.label, hand.bounding_box)
    """

    # MediaPipe's canonical name for each of the 21 landmark indices
    # (index in this list == landmark ID, 0-20). Useful for logging,
    # debugging, and for Phase 2 gesture-rule authoring, e.g.
    # "is INDEX_FINGER_TIP (id 8) above INDEX_FINGER_PIP (id 6)?" -> finger raised.
    LANDMARK_NAMES = [
        "WRIST",                                                                  # 0
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",                        # 1-4
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",   # 5-8
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",  # 9-12
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",  # 13-16
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",                       # 17-20
    ]

    def __init__(self, tracker_config: HandTrackerConfig, display_config: DisplayConfig, logger: logging.Logger):
        self.config = tracker_config
        self.display_config = display_config
        self.logger = logger

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # This single Hands object is created ONCE and reused for every
        # frame (creating it per-frame would be extremely slow, since it
        # loads a TFLite model graph internally).
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.config.static_image_mode,
            max_num_hands=self.config.max_num_hands,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

        self.logger.info(
            "HandTracker initialized (max_hands=%d, model_complexity=%d, "
            "det_conf=%.2f, track_conf=%.2f)",
            self.config.max_num_hands, self.config.model_complexity,
            self.config.min_detection_confidence, self.config.min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> Tuple[List["HandData"], np.ndarray]:
        """
        Runs hand detection + tracking on a single BGR frame.

        Args:
            frame: a BGR image as produced by OpenCV's VideoCapture.

        Returns:
            hands_data: list of HandData, one entry per detected hand
                        (empty list if no hand is visible).
            annotated_frame: a COPY of the input frame with overlays drawn
                        on it (skeleton, labels, bounding boxes, landmark
                        IDs). The original `frame` is left untouched so
                        callers can still access the clean, unmodified
                        image if needed (e.g. for saving raw training data
                        in Phase 2).
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        # MediaPipe's internal model was trained on RGB images, while
        # OpenCV decodes frames as BGR by default -> we must convert.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Marking the array as non-writeable is a documented MediaPipe
        # performance trick: it lets MediaPipe skip an internal defensive
        # copy of the image before running inference.
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hands_data: List[HandData] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            # zip() pairs each hand's 21 landmarks with its corresponding
            # Left/Right classification - MediaPipe guarantees these two
            # lists are always the same length and in the same order.
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label       # "Left" or "Right"
                confidence = handedness.classification[0].score  # 0.0 - 1.0

                landmarks_px: List[Tuple[int, int]] = []
                landmarks_norm: List[Tuple[float, float, float]] = []

                for lm in hand_landmarks.landmark:
                    # lm.x, lm.y are normalized to [0, 1] relative to image
                    # width/height. Multiplying by w/h converts them into
                    # actual pixel coordinates usable by OpenCV drawing calls.
                    px_x = clamp(int(lm.x * w), 0, w - 1)
                    px_y = clamp(int(lm.y * h), 0, h - 1)
                    landmarks_px.append((px_x, px_y))
                    landmarks_norm.append((lm.x, lm.y, lm.z))

                bbox = calculate_bounding_box(landmarks_px)

                hand_data = HandData(
                    label=label,
                    confidence=confidence,
                    landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                    bounding_box=bbox,
                )
                hands_data.append(hand_data)

                self._draw_hand(annotated_frame, hand_landmarks, hand_data, w, h)

        return hands_data, annotated_frame

    def _draw_hand(self, frame: np.ndarray, mp_landmarks, hand_data: HandData, w: int, h: int) -> None:
        """
        Draws all visual overlays for a single hand onto `frame` in place.
        Split out from `process()` to keep each function focused on ONE
        responsibility (detection vs. visualization).
        """

        # 1. Skeleton - draws the 21 landmark dots plus the connecting
        #    lines (e.g. wrist -> thumb, MCP -> PIP -> DIP -> TIP per
        #    finger) using MediaPipe's built-in styled drawer.
        self.mp_drawing.draw_landmarks(
            frame,
            mp_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

        # 2. Bounding box - color-coded by hand label for quick visual
        #    distinction (green = Right, orange = Left).
        if self.display_config.show_bounding_box:
            x_min, y_min, x_max, y_max = hand_data.bounding_box
            x_min, x_max = clamp(x_min, 0, w - 1), clamp(x_max, 0, w - 1)
            y_min, y_max = clamp(y_min, 0, h - 1), clamp(y_max, 0, h - 1)
            color = (0, 255, 0) if hand_data.label == "Right" else (255, 128, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # 3. Handedness label + confidence percentage, drawn above the box.
        if self.display_config.show_handedness_label:
            x_min, y_min, _, _ = hand_data.bounding_box
            label_text = f"{hand_data.label} ({hand_data.confidence * 100:.0f}%)"
            draw_text_with_background(frame, label_text, (max(x_min, 0), max(y_min - 10, 20)))

        # 4. Landmark IDs - verbose per-point numbering, OFF by default
        #    (toggle at runtime with the 'l' key) since it clutters the
        #    view with 21 numbers per hand. Extremely useful for teaching
        #    / debugging which ID corresponds to which physical joint.
        if self.display_config.show_landmark_ids:
            for idx, (px, py) in enumerate(hand_data.landmarks_px):
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (px + 4, py - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        """Releases MediaPipe's internal C++ resources. Call once on app shutdown."""
        self.hands.close()
        self.logger.info("HandTracker closed.")
