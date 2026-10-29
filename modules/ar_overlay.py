"""
modules/ar_overlay.py
========================
Phase 3: the compositor that turns `ui_elements.py` widgets into pixels on the
live camera feed.

HOW THE AR OVERLAY WORKS (per frame, called from app.py)
--------------------------------------------------------
    annotated_frame = ar_overlay.render(annotated_frame, interaction_state)

  1. Get (or reuse) a same-size scratch canvas and clear it (`overlay[:] = 0`).
  2. PASS 1: each visible element advances its animation, then draws ONLY its
     soft panel fill onto the scratch canvas (`draw_panel`).
  3. ONE `cv2.addWeighted(overlay, panel_alpha, frame, 1.0, 0, frame)` blends
     every panel into the frame at once - this is the transparency.
  4. PASS 2: each element draws its crisp parts (borders, text, icons, slider
     fills, knobs) directly on the blended frame (`draw_crisp`). Doing this
     AFTER the blend keeps text fully legible.
  5. Notification toasts are drawn, then the cursor last (always on top).

COORDINATE TRANSFORMATION
-------------------------
Elements keep normalized (0-1) anchors and convert to pixels themselves every
frame (see ui_elements.py), so a new camera resolution rescales the UI
automatically. AROverlay's only resize duty is reallocating its cached scratch
canvas when the frame shape changes.

THEMES
------
One `Theme` object is passed to every draw call; `set_theme()` / `toggle_theme()`
just swap it (elements never cache colors).

OPTIMIZATION
------------
- Scratch canvas allocated once per resolution, reused via `overlay[:] = 0`.
- One addWeighted per frame for all panels.
- Only native OpenCV drawing calls, no per-pixel Python loops.
- O(N) loops over a handful of widgets; notification list is capped.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import AROverlayConfig
from modules.ui_elements import THEMES, Cursor, NotificationToast, Theme, UIElement


class AROverlay:
    """Owns the widgets, theme, notification queue and cursor, and renders them."""

    def __init__(self, config: AROverlayConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.theme: Theme = THEMES.get(config.theme, THEMES["dark"])
        self.elements: Dict[str, UIElement] = {}
        self.notifications: List[NotificationToast] = []
        self.cursor = Cursor()
        self._overlay_buffer: Optional[np.ndarray] = None
        self.logger.info("AROverlay initialized (theme=%s, panel_alpha=%.2f)",
                         self.theme.name, config.panel_alpha)

    # -- elements ------------------------------------------------------
    def add_element(self, element: UIElement) -> None:
        if element.id in self.elements:
            self.logger.warning("Overwriting existing AR element id=%s", element.id)
        self.elements[element.id] = element

    def get_element(self, elem_id: str) -> Optional[UIElement]:
        return self.elements.get(elem_id)

    def remove_element(self, elem_id: str) -> None:
        self.elements.pop(elem_id, None)

    def interactive_elements(self) -> List[UIElement]:
        """Used for collision detection; skips display-only widgets."""
        return [e for e in self.elements.values() if e.visible and e.interactive]

    # -- theme ---------------------------------------------------------
    def set_theme(self, name: str) -> None:
        if name not in THEMES:
            self.logger.warning("Unknown theme '%s'; ignoring.", name)
            return
        self.theme = THEMES[name]
        self.config.theme = name
        self.logger.info("Theme switched to '%s'", name)

    def toggle_theme(self) -> None:
        self.set_theme("light" if self.theme.name == "dark" else "dark")

    # -- notifications -------------------------------------------------
    def add_notification(self, message: str, kind: str = "info", duration: Optional[float] = None) -> None:
        self.notifications.append(
            NotificationToast(message, kind=kind, duration=duration or self.config.notification_duration))
        if len(self.notifications) > self.config.max_notifications:
            self.notifications.pop(0)

    def _prune_expired_notifications(self) -> None:
        self.notifications = [n for n in self.notifications if not n.is_expired()]

    # -- rendering -----------------------------------------------------
    def _get_overlay_buffer(self, frame: np.ndarray) -> np.ndarray:
        if self._overlay_buffer is None or self._overlay_buffer.shape != frame.shape:
            self._overlay_buffer = np.zeros_like(frame)
            self.logger.info("AR overlay buffer (re)allocated for frame shape %s", frame.shape)
        else:
            self._overlay_buffer[:] = 0
        return self._overlay_buffer

    def render(self, frame: np.ndarray, interaction_state=None) -> np.ndarray:
        """Draws everything onto `frame` in place and returns it."""
        if not self.config.show_ar_overlay:
            return frame

        h, w = frame.shape[:2]
        overlay = self._get_overlay_buffer(frame)
        visible = [e for e in self.elements.values() if e.visible]

        # Pass 1: translucent panel fills only
        for element in visible:
            element.update_animation(self.config.hover_animation_speed, self.config.press_animation_speed)
            element.draw_panel(overlay, self.theme, w, h)

        cv2.addWeighted(overlay, self.config.panel_alpha, frame, 1.0, 0, frame)

        # Pass 2: crisp details on the already-blended frame
        for element in visible:
            element.draw_crisp(frame, self.theme, w, h)

        self._prune_expired_notifications()
        for i, toast in enumerate(self.notifications):
            toast.draw(frame, self.theme, w, h, slot=i)

        if interaction_state is not None:
            self.cursor.update(interaction_state.cursor_pos, interaction_state.is_pinching)
            self.cursor.draw(frame, self.theme, dwell_progress=interaction_state.dwell_progress)
        else:
            self.cursor.draw(frame, self.theme)
        return frame
