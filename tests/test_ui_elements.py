"""Unit tests for Phase 3 modules/ui_elements.py (CI-safe: no camera, no MediaPipe)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ui_elements import (
    DARK_THEME, LIGHT_THEME, THEMES, ElementState, Button, ToggleSwitch, Slider,
    VolumeBar, BrightnessBar, StatusIndicator, NotificationToast, Cursor,
    ease_towards, lerp_color,
)


# ---- coordinates / collision ----
def test_pixel_rect_conversion():
    b = Button("b", 0.25, 0.5, 0.1, 0.2, label="X")
    assert b.get_pixel_rect(1000, 800) == (250, 400, 100, 160)


def test_pixel_rect_scales_with_resolution():
    b = Button("b", 0.25, 0.5, 0.1, 0.2, label="X")
    small, large = b.get_pixel_rect(640, 480), b.get_pixel_rect(1280, 960)
    assert large[0] == small[0] * 2 and large[2] == small[2] * 2


def test_contains_point_inside_and_edges():
    b = Button("b", 0.1, 0.1, 0.2, 0.2, label="X")
    assert b.contains_point(200, 200, 1000, 1000)
    assert b.contains_point(100, 100, 1000, 1000)
    assert b.contains_point(300, 300, 1000, 1000)


def test_contains_point_outside():
    b = Button("b", 0.1, 0.1, 0.2, 0.2, label="X")
    assert not b.contains_point(50, 50, 1000, 1000)
    assert not b.contains_point(301, 301, 1000, 1000)


# ---- easing ----
def test_ease_moves_toward_target():
    assert 0.0 < ease_towards(0.0, 1.0, 0.1, 8.0) < 1.0


def test_ease_zero_dt_no_move():
    assert ease_towards(0.3, 1.0, 0.0) == 0.3


def test_ease_converges():
    v = 0.0
    for _ in range(200):
        v = ease_towards(v, 1.0, 0.05, 8.0)
    assert v == pytest.approx(1.0, abs=0.01)


def test_lerp_color():
    assert lerp_color((0, 0, 0), (100, 200, 50), 0.5) == (50, 100, 25)
    assert lerp_color((0, 0, 0), (100, 100, 100), 1.5) == (100, 100, 100)
    assert lerp_color((0, 0, 0), (100, 100, 100), -1) == (0, 0, 0)


# ---- animation state ----
def test_hover_progress_rises_when_hovered():
    b = Button("b", 0, 0, 0.1, 0.1, label="X")
    b.state = ElementState.HOVER
    time.sleep(0.02)
    b.update_animation()
    assert b._hover_progress > 0.0


def test_hover_progress_falls_when_idle():
    b = Button("b", 0, 0, 0.1, 0.1, label="X")
    b._hover_progress = 1.0
    b._last_update_time = time.time() - 0.5
    b.update_animation()
    assert b._hover_progress < 1.0


def test_trigger_pulse():
    b = Button("b", 0, 0, 0.1, 0.1, label="X")
    b.trigger_pulse()
    assert b._pulse_progress == 1.0


# ---- toggle ----
def test_toggle_flips():
    t = ToggleSwitch("t", 0, 0, 0.1, 0.05, value=False, label="X")
    t.toggle()
    assert t.value is True
    t.toggle()
    assert t.value is False


def test_toggle_knob_animates():
    """Regression test: knob dt used to be ~0 (base class reset the timestamp first)."""
    t = ToggleSwitch("t", 0, 0, 0.1, 0.05, value=False, label="X")
    t.value = True
    t._last_update_time = time.time() - 0.5
    t.update_animation()
    assert t._knob_progress > 0.0


# ---- sliders ----
def test_slider_value_clamped():
    assert Slider("s", 0, 0, 0.1, 0.1, value=1.5).value == 1.0
    assert Slider("s", 0, 0, 0.1, 0.1, value=-0.3).value == 0.0


def test_slider_horizontal_pointer():
    s = Slider("s", 0.1, 0.1, 0.2, 0.05, vertical=False)
    s.set_value_from_pointer((100, 125), 1000, 1000)
    assert s.value == pytest.approx(0.0, abs=0.01)
    s.set_value_from_pointer((300, 125), 1000, 1000)
    assert s.value == pytest.approx(1.0, abs=0.01)
    s.set_value_from_pointer((200, 125), 1000, 1000)
    assert s.value == pytest.approx(0.5, abs=0.01)


def test_slider_vertical_fills_bottom_up():
    s = Slider("s", 0.1, 0.1, 0.05, 0.3, vertical=True)
    s.set_value_from_pointer((125, 400), 1000, 1000)
    assert s.value == pytest.approx(0.0, abs=0.01)
    s.set_value_from_pointer((125, 100), 1000, 1000)
    assert s.value == pytest.approx(1.0, abs=0.01)


def test_slider_clamps_outside_track():
    s = Slider("s", 0.1, 0.1, 0.2, 0.05)
    s.set_value_from_pointer((-500, 125), 1000, 1000)
    assert s.value == 0.0
    s.set_value_from_pointer((5000, 125), 1000, 1000)
    assert s.value == 1.0


def test_volume_and_brightness_bars():
    v = VolumeBar("v", 0.8, 0.1, 0.04, 0.3, value=0.7)
    assert v.vertical and v.icon == "volume" and v.value == 0.7
    b = BrightnessBar("b", 0.9, 0.1, 0.04, 0.3, value=0.4)
    assert b.vertical and b.icon == "brightness"


# ---- status / toast / cursor ----
def test_status_indicator_not_interactive_by_default():
    assert StatusIndicator("s", 0, 0, 0.1, 0.05, value="A").interactive is False
    assert StatusIndicator("s", 0, 0, 0.1, 0.05, value="A", interactive=True).interactive is True


def test_notification_lifecycle():
    t = NotificationToast("Hi", duration=2.0)
    assert not t.is_expired()
    t.created_at = time.time() - 1.0
    assert t.alpha() == pytest.approx(1.0, abs=0.01)
    t.created_at = time.time() - 1.9
    assert 0.0 < t.alpha() < 1.0
    t.created_at = time.time() - 3.0
    assert t.is_expired()


def test_cursor_stores_state_and_noop_without_position():
    c = Cursor()
    c.update((10, 20), True)
    assert c.position == (10, 20) and c.is_pinching
    c2 = Cursor()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    c2.draw(frame, DARK_THEME)
    assert frame.sum() == 0


def test_theme_registry():
    assert THEMES["dark"] is DARK_THEME and THEMES["light"] is LIGHT_THEME
    assert DARK_THEME.panel_bg != LIGHT_THEME.panel_bg


def test_two_pass_draw_keeps_light_theme_text_dark():
    """Regression test: crisp text must be drawn AFTER the panel blend."""
    import logging
    from config import AROverlayConfig
    from modules.ar_overlay import AROverlay
    cfg = AROverlayConfig(theme="light")
    overlay = AROverlay(cfg, logging.getLogger("t"))
    overlay.add_element(Button("b", 0.1, 0.1, 0.4, 0.2, label="Start"))
    frame = np.full((400, 600, 3), (60, 55, 50), dtype=np.uint8)
    overlay.render(frame)
    x, y, w, h = overlay.get_element("b").get_pixel_rect(600, 400)
    region = frame[y:y + h, x:x + w]
    assert region.reshape(-1, 3).min(axis=0).max() < 100  # dark text pixels exist
