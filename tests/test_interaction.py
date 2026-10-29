"""Unit tests for Phase 3 modules/interaction.py (CI-safe, hand data built by hand)."""
import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AROverlayConfig
from modules.ar_overlay import AROverlay
from modules.gesture_recognition import Gesture, GestureResult, FingerStates, INDEX_TIP
from modules.hand_tracker import HandData
from modules.interaction import PointerTracker, InteractionEngine
from modules.ui_elements import Button, Slider, StatusIndicator, ElementState

W, H = 1000, 1000


def make_hand(tip):
    pts = [(0, 0)] * 21
    pts[INDEX_TIP] = tip
    return HandData("Right", 0.95, pts, [(x / W, y / H, 0.0) for x, y in pts], (0, 0, W, H))


def make_gesture(g):
    return GestureResult(g, 0.9, FingerStates(), 0, True, g)


def make_overlay(**kw):
    logger = logging.getLogger("test_interaction")
    logger.addHandler(logging.NullHandler())
    cfg = AROverlayConfig(**kw)
    return AROverlay(cfg, logger), cfg, logger


def center(el):
    x, y, w, h = el.get_pixel_rect(W, H)
    return (x + w // 2, y + h // 2)


def settle(engine, overlay, hand, gesture, n=20):
    s = None
    for _ in range(n):
        s = engine.update(overlay, hand, make_gesture(gesture), W, H)
    return s


# ---- pointer tracker ----
def test_pointer_snaps_first_reading():
    assert PointerTracker(0.5).update((100, 200)) == (100, 200)


def test_pointer_smooths():
    t = PointerTracker(0.5)
    t.update((0, 0))
    assert t.update((100, 100)) == (50, 50)


def test_pointer_converges():
    t = PointerTracker(0.5)
    t.update((0, 0))
    for _ in range(20):
        pos = t.update((100, 100))
    assert pos[0] == pytest.approx(100, abs=1) and pos[1] == pytest.approx(100, abs=1)


def test_pointer_reset_and_none():
    t = PointerTracker(0.5)
    t.update((50, 50))
    t.reset()
    assert t.update((200, 200)) == (200, 200)
    assert t.update(None) is None


# ---- hover / collision ----
def test_hover_detected():
    o, c, l = make_overlay()
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    o.add_element(b)
    s = InteractionEngine(c, l).update(o, make_hand(center(b)), make_gesture(Gesture.OPEN_PALM), W, H)
    assert s.hovered_id == "b" and b.state == ElementState.HOVER


def test_no_hover_elsewhere():
    o, c, l = make_overlay()
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    o.add_element(b)
    s = InteractionEngine(c, l).update(o, make_hand((900, 900)), make_gesture(Gesture.OPEN_PALM), W, H)
    assert s.hovered_id is None and b.state == ElementState.IDLE


def test_noninteractive_never_hovered():
    o, c, l = make_overlay()
    st = StatusIndicator("s", 0.1, 0.1, 0.2, 0.1, value="X")
    o.add_element(st)
    s = InteractionEngine(c, l).update(o, make_hand(center(st)), make_gesture(Gesture.OPEN_PALM), W, H)
    assert s.hovered_id is None


# ---- virtual click ----
def test_click_on_pinch_down_up():
    o, c, l = make_overlay()
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    clicked = []
    b.on_click = lambda e: clicked.append(e.id)
    o.add_element(b)
    e = InteractionEngine(c, l)
    hand = make_hand(center(b))
    e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    s = e.update(o, hand, make_gesture(Gesture.PINCH), W, H)
    assert s.pressed_id == "b" and clicked == []
    s = e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    assert s.selected_id == "b" and clicked == ["b"]
    assert b._pulse_progress == 1.0


def test_click_cancelled_when_released_elsewhere():
    o, c, l = make_overlay()
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    clicked = []
    b.on_click = lambda e: clicked.append(e.id)
    o.add_element(b)
    e = InteractionEngine(c, l)
    on, away = make_hand(center(b)), make_hand((900, 900))
    e.update(o, on, make_gesture(Gesture.OPEN_PALM), W, H)
    e.update(o, on, make_gesture(Gesture.PINCH), W, H)
    settle(e, o, away, Gesture.PINCH, 20)
    s = e.update(o, away, make_gesture(Gesture.OPEN_PALM), W, H)
    assert s.selected_id is None and clicked == []


# ---- dwell ----
def test_dwell_progress_grows():
    o, c, l = make_overlay(dwell_seconds=0.2)
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    o.add_element(b)
    e = InteractionEngine(c, l)
    hand = make_hand(center(b))
    s1 = e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    time.sleep(0.1)
    s2 = e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    assert s2.dwell_progress > s1.dwell_progress


def test_dwell_selection_fires():
    o, c, l = make_overlay(dwell_seconds=0.05)
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    clicked = []
    b.on_click = lambda e: clicked.append(e.id)
    o.add_element(b)
    e = InteractionEngine(c, l)
    hand = make_hand(center(b))
    for _ in range(10):
        s = e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
        if s.selected_id:
            break
        time.sleep(0.02)
    assert s.selected_id == "b" and clicked == ["b"]


def test_dwell_restarts_on_new_element():
    o, c, l = make_overlay(dwell_seconds=5.0)
    b1 = Button("b1", 0.0, 0.0, 0.2, 0.1, label="A")
    b2 = Button("b2", 0.5, 0.5, 0.2, 0.1, label="B")
    o.add_element(b1)
    o.add_element(b2)
    e = InteractionEngine(c, l)
    e.update(o, make_hand(center(b1)), make_gesture(Gesture.OPEN_PALM), W, H)
    time.sleep(0.05)
    s = settle(e, o, make_hand(center(b2)), Gesture.OPEN_PALM, 20)  # pointer is smoothed: needs frames to travel
    assert s.dwell_target_id == "b2" and s.dwell_progress < 0.1


def test_no_dwell_while_pinching():
    o, c, l = make_overlay(dwell_seconds=0.05)
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    o.add_element(b)
    e = InteractionEngine(c, l)
    hand = make_hand(center(b))
    e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    time.sleep(0.1)
    assert e.update(o, hand, make_gesture(Gesture.PINCH), W, H).dwell_progress == 0.0


# ---- slider drag ----
def test_pinch_drag_updates_slider():
    o, c, l = make_overlay(pinch_drag_enabled=True)
    s = Slider("s", 0.1, 0.1, 0.2, 0.05, value=0.0)
    o.add_element(s)
    e = InteractionEngine(c, l)
    x, y, w, h = s.get_pixel_rect(W, H)
    start = make_hand((x + 5, y + h // 2))
    e.update(o, start, make_gesture(Gesture.OPEN_PALM), W, H)
    e.update(o, start, make_gesture(Gesture.PINCH), W, H)
    settle(e, o, make_hand((x + w - 5, y + h // 2)), Gesture.PINCH, 20)
    assert s.value > 0.8


def test_pinch_drag_disabled():
    o, c, l = make_overlay(pinch_drag_enabled=False)
    s = Slider("s", 0.1, 0.1, 0.2, 0.05, value=0.3)
    o.add_element(s)
    e = InteractionEngine(c, l)
    x, y, w, h = s.get_pixel_rect(W, H)
    hand = make_hand((x + w - 5, y + h // 2))
    e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    settle(e, o, hand, Gesture.PINCH, 3)
    assert s.value == 0.3


# ---- hand lost ----
def test_hand_lost_clears_state():
    o, c, l = make_overlay()
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    o.add_element(b)
    e = InteractionEngine(c, l)
    hand = make_hand(center(b))
    e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    e.update(o, hand, make_gesture(Gesture.PINCH), W, H)
    s = e.update(o, None, None, W, H)
    assert s.cursor_pos is None and s.hovered_id is None and s.pressed_id is None
    assert b.state == ElementState.IDLE


def test_no_stale_click_after_hand_lost():
    o, c, l = make_overlay()
    b = Button("b", 0.1, 0.1, 0.2, 0.1, label="X")
    clicked = []
    b.on_click = lambda e: clicked.append(e.id)
    o.add_element(b)
    e = InteractionEngine(c, l)
    hand = make_hand(center(b))
    e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    e.update(o, hand, make_gesture(Gesture.PINCH), W, H)
    e.update(o, None, None, W, H)
    s = e.update(o, hand, make_gesture(Gesture.OPEN_PALM), W, H)
    assert s.selected_id is None and clicked == []
