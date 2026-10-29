"""
modules/interaction.py
=========================
Phase 3: connects Phase 2's gesture output to the AR widgets.

THE INTERACTION ENGINE (per frame)
----------------------------------
    interaction_state = interaction_engine.update(
        ar_overlay, primary_hand, gesture_result, frame_w, frame_h)

  1. POINTER TRACKING - the INDEX_TIP pixel position from Phase 1's HandData,
     smoothed with an exponential moving average (same idea as fps.py):
         smoothed = smoothed * a + raw * (1 - a)
  2. COLLISION DETECTION - the smoothed cursor is point-tested against every
     interactive element's bounding box; the first hit is "hovered".
  3. VIRTUAL CLICK - Phase 2's Gesture.PINCH is the "mouse button". Pinch START
     over an element = PRESSED. Pinch END over the SAME element = click. If the
     cursor leaves mid-pinch, the press is cancelled (no click).
  4. SELECTION TIMER (dwell) - hovering WITHOUT pinching for `dwell_seconds`
     also clicks. dwell_progress = elapsed / dwell_seconds drives the arc drawn
     around the cursor. Leaving the element resets the timer.
  5. SLIDER DRAGGING - while pinch-held on a Slider/VolumeBar/BrightnessBar,
     its value follows the cursor.

Nothing here touches MediaPipe or re-implements gesture classification.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import AROverlayConfig
from modules.ar_overlay import AROverlay
from modules.gesture_recognition import Gesture, GestureResult, INDEX_TIP
from modules.hand_tracker import HandData
from modules.ui_elements import ElementState, Slider


class PointerTracker:
    """EMA-smoothed cursor position driven by the index fingertip."""

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing
        self._smoothed: Optional[Tuple[float, float]] = None

    def update(self, raw_pos: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if raw_pos is None:
            self._smoothed = None
            return None
        if self._smoothed is None:
            self._smoothed = (float(raw_pos[0]), float(raw_pos[1]))
        else:
            a = self.smoothing
            self._smoothed = (self._smoothed[0] * a + raw_pos[0] * (1 - a),
                              self._smoothed[1] * a + raw_pos[1] * (1 - a))
        return (int(self._smoothed[0]), int(self._smoothed[1]))

    def reset(self) -> None:
        self._smoothed = None


@dataclass
class InteractionState:
    """Everything render() and app.py need for one frame."""
    cursor_pos: Optional[Tuple[int, int]]
    is_pinching: bool
    hovered_id: Optional[str]
    pressed_id: Optional[str]
    selected_id: Optional[str]      # set for exactly the one frame a click fires
    dwell_progress: float           # 0.0-1.0
    dwell_target_id: Optional[str]


class InteractionEngine:
    """Per-frame hover / press / release / dwell / drag logic."""

    def __init__(self, config: AROverlayConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.pointer = PointerTracker(smoothing=config.pointer_smoothing)
        self._pressed_id: Optional[str] = None
        self._was_pinching = False
        self._hover_start_time: Dict[str, float] = {}

    def reset(self) -> None:
        """Clear all press/hover/dwell state (called when the hand is lost)."""
        self.pointer.reset()
        self._pressed_id = None
        self._was_pinching = False
        self._hover_start_time.clear()

    def update(self, overlay: AROverlay, hand_data: Optional[HandData],
               gesture_result: Optional[GestureResult], frame_w: int, frame_h: int) -> InteractionState:
        if hand_data is None:
            self.reset()
            for element in overlay.elements.values():
                element.state = ElementState.IDLE
            return InteractionState(None, False, None, None, None, 0.0, None)

        # 1. pointer tracking
        tip = hand_data.landmarks_px[INDEX_TIP]
        cursor_pos = self.pointer.update((tip[0], tip[1]))
        is_pinching = bool(gesture_result and gesture_result.gesture == Gesture.PINCH)

        # 2. collision detection
        hovered_id = None
        if cursor_pos is not None:
            for element in overlay.interactive_elements():
                if element.contains_point(cursor_pos[0], cursor_pos[1], frame_w, frame_h):
                    hovered_id = element.id
                    break

        # 3. virtual click
        selected_id = None
        pinch_started = is_pinching and not self._was_pinching
        pinch_ended = (not is_pinching) and self._was_pinching
        if pinch_started and hovered_id is not None:
            self._pressed_id = hovered_id
        if pinch_ended:
            if self._pressed_id is not None and self._pressed_id == hovered_id:
                selected_id = self._pressed_id
            self._pressed_id = None

        # 4. dwell selection timer (only while not pinching)
        dwell_progress, dwell_target_id = 0.0, None
        if hovered_id is not None and not is_pinching:
            start = self._hover_start_time.setdefault(hovered_id, time.time())
            dwell_progress = min((time.time() - start) / max(self.config.dwell_seconds, 1e-6), 1.0)
            dwell_target_id = hovered_id
            if dwell_progress >= 1.0:
                selected_id = hovered_id
                self._hover_start_time.pop(hovered_id, None)
            for other in list(self._hover_start_time):
                if other != hovered_id:
                    self._hover_start_time.pop(other, None)
        else:
            self._hover_start_time.clear()

        # 5. slider dragging
        if self.config.pinch_drag_enabled and is_pinching and self._pressed_id and cursor_pos is not None:
            pressed = overlay.get_element(self._pressed_id)
            if isinstance(pressed, Slider):
                pressed.set_value_from_pointer(cursor_pos, frame_w, frame_h)

        # visual states
        for element in overlay.elements.values():
            if not element.interactive:
                continue
            if element.id == hovered_id and is_pinching:
                element.state = ElementState.PRESSED
            elif element.id == hovered_id:
                element.state = ElementState.HOVER
            else:
                element.state = ElementState.IDLE

        # fire click exactly once
        if selected_id is not None:
            element = overlay.get_element(selected_id)
            if element is not None:
                element.trigger_pulse()
                if element.on_click is not None:
                    element.on_click(element)
                self.logger.info("AR element selected: %s", selected_id)

        self._was_pinching = is_pinching
        return InteractionState(cursor_pos, is_pinching, hovered_id, self._pressed_id,
                                selected_id, dwell_progress, dwell_target_id)
