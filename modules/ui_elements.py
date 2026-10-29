"""
modules/ui_elements.py
========================
Phase 3: the visual widgets of the floating AR interface - buttons, sliders,
toggle switches, volume/brightness bars, cursor, status badges and
notification toasts.

This module answers one question: WHAT does each widget look like, and what
state is it in? It does not decide where widgets are composited (ar_overlay.py)
or when they get clicked (interaction.py).

COORDINATE SYSTEM
-----------------
Every element stores position and size as NORMALIZED floats in [0, 1]:

    pixel_x = anchor_x * frame_width      pixel_w = width  * frame_width
    pixel_y = anchor_y * frame_height     pixel_h = height * frame_height

The conversion is redone every frame in `get_pixel_rect()`, so changing the
camera resolution rescales the whole interface automatically (this is the
"resizable interface"). It is the same scale-invariance idea Phase 2 uses
with palm-size ratios.

TWO-PASS DRAWING
----------------
Each element draws in two steps, with the translucent blend in between:

    pass 1  draw_panel(overlay, ...)  soft background fill only
    [ cv2.addWeighted blends the overlay canvas into the frame ]
    pass 2  draw_crisp(frame, ...)    borders, text, icons, value fills, knobs

Rule of thumb: anything that conveys STATE or must stay readable goes in
draw_crisp(); purely decorative background goes in draw_panel(). Drawing text
BEFORE the blend would let the translucent panel paint over it and wash it out.

ANIMATION
---------
Hover/press/pulse values ease toward their targets with a frame-rate
independent exponential:  progress += (target - progress) * (1 - exp(-speed*dt))
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Themes (BGR colors)
# ---------------------------------------------------------------------------

@dataclass
class Theme:
    name: str
    panel_bg: Tuple[int, int, int]
    accent: Tuple[int, int, int]
    accent_dim: Tuple[int, int, int]
    text: Tuple[int, int, int]
    text_dim: Tuple[int, int, int]
    border: Tuple[int, int, int]
    success: Tuple[int, int, int]
    warning: Tuple[int, int, int]
    danger: Tuple[int, int, int]


DARK_THEME = Theme(
    name="dark", panel_bg=(40, 32, 28), accent=(230, 170, 60), accent_dim=(150, 100, 40),
    text=(245, 245, 245), text_dim=(170, 170, 170), border=(90, 80, 75),
    success=(110, 200, 80), warning=(50, 180, 240), danger=(70, 70, 230),
)

LIGHT_THEME = Theme(
    name="light", panel_bg=(235, 235, 238), accent=(200, 120, 20), accent_dim=(220, 180, 140),
    text=(35, 35, 35), text_dim=(110, 110, 110), border=(190, 190, 190),
    success=(90, 170, 60), warning=(30, 150, 220), danger=(50, 50, 200),
)

THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_rounded_rect(img, x, y, w, h, color, radius=10, filled=True, thickness=2):
    """OpenCV has no rounded rectangle: build one from rectangles + corner arcs."""
    r = max(0, min(radius, w // 2, h // 2))
    if filled:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness)


def lerp_color(c1, c2, t):
    """Linear interpolation between two BGR colors (t clamped to [0, 1])."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def ease_towards(current, target, dt, speed=8.0):
    """Frame-rate-independent exponential approach toward target."""
    if dt <= 0:
        return current
    return current + (target - current) * (1 - math.exp(-speed * dt))


def centered_text_origin(text, cx, cy, font, scale, thickness):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return cx - tw // 2, cy + th // 2


class ElementState(Enum):
    IDLE = "idle"
    HOVER = "hover"
    PRESSED = "pressed"
    DISABLED = "disabled"


# ---------------------------------------------------------------------------
# Base element
# ---------------------------------------------------------------------------

class UIElement:
    """Normalized geometry, hit-testing, animation state and color helpers."""

    def __init__(self, elem_id: str, anchor_x: float, anchor_y: float, width: float,
                 height: float, visible: bool = True, interactive: bool = True, label: str = ""):
        self.id = elem_id
        self.anchor_x, self.anchor_y = anchor_x, anchor_y
        self.width, self.height = width, height
        self.visible = visible
        self.interactive = interactive
        self.label = label
        self.state = ElementState.IDLE
        self.on_click: Optional[Callable[["UIElement"], None]] = None
        self._hover_progress = 0.0
        self._press_progress = 0.0
        self._pulse_progress = 0.0
        self._last_update_time = time.time()

    def get_pixel_rect(self, frame_w: int, frame_h: int):
        return (int(self.anchor_x * frame_w), int(self.anchor_y * frame_h),
                int(self.width * frame_w), int(self.height * frame_h))

    def contains_point(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        """Axis-aligned bounding-box point test (collision detection). The
        fingertip cursor is treated as a single point, like a mouse pointer."""
        x, y, w, h = self.get_pixel_rect(frame_w, frame_h)
        return x <= px <= x + w and y <= py <= y + h

    def update_animation(self, hover_speed: float = 8.0, press_speed: float = 14.0) -> None:
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now
        hover_target = 1.0 if self.state in (ElementState.HOVER, ElementState.PRESSED) else 0.0
        self._hover_progress = ease_towards(self._hover_progress, hover_target, dt, hover_speed)
        press_target = 1.0 if self.state == ElementState.PRESSED else 0.0
        self._press_progress = ease_towards(self._press_progress, press_target, dt, press_speed)
        self._pulse_progress = ease_towards(self._pulse_progress, 0.0, dt, speed=5.0)

    def trigger_pulse(self) -> None:
        self._pulse_progress = 1.0

    def draw_panel(self, overlay, theme, frame_w, frame_h) -> None:
        raise NotImplementedError

    def draw_crisp(self, frame, theme, frame_w, frame_h) -> None:
        raise NotImplementedError

    def _panel_color(self, theme):
        mix = lerp_color(theme.panel_bg, theme.accent_dim, 0.5 * self._hover_progress)
        return lerp_color(mix, theme.accent, 0.5 * self._press_progress)

    def _border_color(self, theme):
        return lerp_color(theme.border, theme.accent, max(self._hover_progress, self._press_progress))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class Button(UIElement):
    """Floating button: grows on hover, tints on press, pulse ring on click."""

    def _grown_rect(self, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        g = int(3 * self._hover_progress)
        return x - g, y - g, w + 2 * g, h + 2 * g

    def draw_panel(self, overlay, theme, fw, fh):
        rx, ry, rw, rh = self._grown_rect(fw, fh)
        draw_rounded_rect(overlay, rx, ry, rw, rh, self._panel_color(theme), radius=12)

    def draw_crisp(self, frame, theme, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        rx, ry, rw, rh = self._grown_rect(fw, fh)
        draw_rounded_rect(frame, rx, ry, rw, rh, self._border_color(theme), radius=12, filled=False, thickness=2)
        color = theme.text if self.state != ElementState.DISABLED else theme.text_dim
        tx, ty = centered_text_origin(self.label, x + w // 2, y + h // 2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if self._pulse_progress > 0.01:
            r = int(max(w, h) * 0.55 * (1.5 - self._pulse_progress))
            cv2.circle(frame, (x + w // 2, y + h // 2), max(r, 1),
                       lerp_color(theme.panel_bg, theme.success, self._pulse_progress), 2)


class ToggleSwitch(UIElement):
    """On/off pill switch with an animated knob."""

    def __init__(self, *args, value: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value
        self._knob_progress = 1.0 if value else 0.0

    def toggle(self) -> None:
        self.value = not self.value

    def update_animation(self, hover_speed: float = 8.0, press_speed: float = 14.0) -> None:
        # Capture dt BEFORE super() resets _last_update_time, otherwise dt is
        # always ~0 and the knob never visibly animates.
        dt = max(time.time() - self._last_update_time, 0.0)
        super().update_animation(hover_speed, press_speed)
        self._knob_progress = ease_towards(self._knob_progress, 1.0 if self.value else 0.0, dt, speed=10.0)

    def draw_panel(self, overlay, theme, fw, fh):
        pass  # the track color IS the on/off indicator, so it is drawn opaque in draw_crisp

    def draw_crisp(self, frame, theme, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        track = lerp_color(theme.border, theme.success, self._knob_progress)
        track = lerp_color(track, theme.accent, 0.4 * self._hover_progress)
        draw_rounded_rect(frame, x, y, w, h, track, radius=h // 2)
        kr = int(h * 0.42)
        kx = int(x + kr + 2 + (w - 2 * kr - 4) * self._knob_progress)
        ky = y + h // 2
        cv2.circle(frame, (kx, ky), kr, theme.text, -1, cv2.LINE_AA)
        cv2.circle(frame, (kx, ky), kr, self._border_color(theme), 1, cv2.LINE_AA)
        if self.label:
            cv2.putText(frame, self.label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme.text_dim, 1, cv2.LINE_AA)


class Slider(UIElement):
    """Draggable value bar; value is normalized to [0, 1]."""

    def __init__(self, *args, value: float = 0.5, vertical: bool = False,
                 fill_color=None, icon: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.value = max(0.0, min(1.0, value))
        self.vertical = vertical
        self.fill_color = fill_color
        self.icon = icon

    def set_value_from_pointer(self, cursor_px, frame_w: int, frame_h: int) -> None:
        """Project the cursor onto the track axis (vertical bars fill bottom-up)."""
        x, y, w, h = self.get_pixel_rect(frame_w, frame_h)
        t = (y + h - cursor_px[1]) / max(h, 1) if self.vertical else (cursor_px[0] - x) / max(w, 1)
        self.value = max(0.0, min(1.0, t))

    def draw_panel(self, overlay, theme, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        draw_rounded_rect(overlay, x, y, w, h, self._panel_color(theme), radius=8)

    def draw_crisp(self, frame, theme, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        fill = self.fill_color or theme.accent
        draw_rounded_rect(frame, x, y, w, h, self._border_color(theme), radius=8, filled=False, thickness=2)
        pad = 4
        if self.vertical:
            fh_ = int((h - 2 * pad) * self.value)
            fx, fy, fw_, fh2 = x + pad, y + h - pad - fh_, w - 2 * pad, fh_
        else:
            fw2 = int((w - 2 * pad) * self.value)
            fx, fy, fw_, fh2 = x + pad, y + pad, fw2, h - 2 * pad
        if fw_ > 0 and fh2 > 0:
            draw_rounded_rect(frame, fx, fy, max(fw_, 4), max(fh2, 4), fill, radius=6)
        if self.icon == "volume":
            cx, cy = x + w // 2, y - 22
            pts = np.array([[cx - 8, cy - 6], [cx - 8, cy + 6], [cx, cy + 10], [cx, cy - 10]], dtype=np.int32)
            cv2.fillConvexPoly(frame, pts, theme.text_dim)
            cv2.rectangle(frame, (cx - 12, cy - 4), (cx - 8, cy + 4), theme.text_dim, -1)
        elif self.icon == "brightness":
            cx, cy = x + w // 2, y - 22
            cv2.circle(frame, (cx, cy), 6, theme.text_dim, -1, cv2.LINE_AA)
            for dx, dy in [(0, -11), (0, 11), (-11, 0), (11, 0)]:
                cv2.line(frame, (cx, cy), (cx + dx, cy + dy), theme.text_dim, 1, cv2.LINE_AA)
        if self.label:
            pos = (x - 2, y + h + 18) if self.vertical else (x, y - 8)
            cv2.putText(frame, self.label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme.text_dim, 1, cv2.LINE_AA)
        pct = f"{int(self.value * 100)}%"
        pos = (x - 4, y - 8) if self.vertical else (x + w + 8, y + h - 6)
        cv2.putText(frame, pct, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme.text, 1, cv2.LINE_AA)


class VolumeBar(Slider):
    """Pre-styled vertical slider for volume."""
    def __init__(self, elem_id, anchor_x, anchor_y, width, height, value=0.5, **kwargs):
        super().__init__(elem_id, anchor_x, anchor_y, width, height, value=value,
                          vertical=True, icon="volume", label=kwargs.pop("label", "Vol"), **kwargs)


class BrightnessBar(Slider):
    """Pre-styled vertical slider for brightness."""
    def __init__(self, elem_id, anchor_x, anchor_y, width, height, value=0.5, **kwargs):
        super().__init__(elem_id, anchor_x, anchor_y, width, height, value=value,
                          vertical=True, icon="brightness", label=kwargs.pop("label", "Bright"), **kwargs)


class StatusIndicator(UIElement):
    """Read-only pill badge, e.g. 'Tracking: Active'. Not interactive."""

    def __init__(self, *args, value: str = "", status: str = "neutral", **kwargs):
        kwargs.setdefault("interactive", False)
        super().__init__(*args, **kwargs)
        self.value = value
        self.status = status  # good | warning | bad | neutral

    def draw_panel(self, overlay, theme, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        draw_rounded_rect(overlay, x, y, w, h, theme.panel_bg, radius=h // 2)

    def draw_crisp(self, frame, theme, fw, fh):
        x, y, w, h = self.get_pixel_rect(fw, fh)
        dot = {"good": theme.success, "warning": theme.warning, "bad": theme.danger}.get(self.status, theme.text_dim)
        draw_rounded_rect(frame, x, y, w, h, theme.border, radius=h // 2, filled=False, thickness=1)
        r = max(h // 6, 3)
        cv2.circle(frame, (x + r + 8, y + h // 2), r, dot, -1, cv2.LINE_AA)
        text = f"{self.label}: {self.value}" if self.label else self.value
        cv2.putText(frame, text, (x + 2 * r + 14, y + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    theme.text, 1, cv2.LINE_AA)


class NotificationToast:
    """Short-lived message: fades in, holds, fades out, then expires."""

    FADE_IN_S = 0.2
    FADE_OUT_S = 0.35

    def __init__(self, message: str, kind: str = "info", duration: float = 2.5):
        self.message, self.kind, self.duration = message, kind, duration
        self.created_at = time.time()

    def alpha(self) -> float:
        age = time.time() - self.created_at
        if age < self.FADE_IN_S:
            return age / self.FADE_IN_S
        if age > self.duration - self.FADE_OUT_S:
            return max(0.0, (self.duration - age) / self.FADE_OUT_S)
        return 1.0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.duration

    def draw(self, frame, theme, frame_w, frame_h, slot: int) -> None:
        a = self.alpha()
        if a <= 0.01:
            return
        color = {"info": theme.accent, "success": theme.success, "warning": theme.warning}.get(self.kind, theme.accent)
        w, h = 260, 44
        x = frame_w - w - 20
        slide = int((1 - a) * 30)
        y = 20 + slot * (h + 10)
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        draw_rounded_rect(panel, 0, 0, w, h, theme.panel_bg, radius=10)
        roi = frame[y:y + h, x + slide:x + slide + w]
        if roi.shape[:2] == (h, w):
            frame[y:y + h, x + slide:x + slide + w] = cv2.addWeighted(panel, 0.85 * a, roi, 1 - 0.85 * a, 0)
        cv2.rectangle(frame, (x + slide, y), (x + slide + 4, y + h), color, -1)
        cv2.putText(frame, self.message, (x + slide + 16, y + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    lerp_color(theme.panel_bg, theme.text, a), 1, cv2.LINE_AA)


class Cursor:
    """On-screen pointer following the fingertip; ring tightens while pinching
    and a green arc sweeps during a dwell-timer selection."""

    def __init__(self):
        self.position: Optional[Tuple[int, int]] = None
        self.is_pinching = False
        self._ring_progress = 0.0
        self._last_update_time = time.time()

    def update(self, position, is_pinching: bool) -> None:
        self.position = position
        self.is_pinching = is_pinching

    def draw(self, frame, theme, dwell_progress: float = 0.0) -> None:
        if self.position is None:
            return
        now = time.time()
        dt = max(now - self._last_update_time, 0.0)
        self._last_update_time = now
        self._ring_progress = ease_towards(self._ring_progress, 1.0 if self.is_pinching else 0.0, dt, 12.0)
        px, py = int(self.position[0]), int(self.position[1])
        outer = int(16 - 4 * self._ring_progress)
        color = lerp_color(theme.text, theme.accent, self._ring_progress)
        cv2.circle(frame, (px, py), outer, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3 + int(3 * self._ring_progress), color, -1, cv2.LINE_AA)
        if dwell_progress > 0.01:
            cv2.ellipse(frame, (px, py), (outer + 6, outer + 6), -90, 0, int(360 * dwell_progress),
                        theme.success, 2, cv2.LINE_AA)
