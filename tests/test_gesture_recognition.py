"""
tests/test_gesture_recognition.py
===================================
Unit tests for Phase 2 (modules/gesture_recognition.py).

Design note, following the same philosophy as Phase 1's tests/test_basic.py:
these tests are CI-safe (no webcam, no MediaPipe inference) because they
build synthetic 21-point landmark sets by hand instead of capturing real
frames. Every synthetic point below is deliberately chosen and verified
against the underlying angle/distance formulas (see the comments next to
each constant) rather than eyeballed, so a failing test points at a real
regression rather than an unrealistic fixture.

All synthetic hands share one coordinate scheme so the numbers are easy to
cross-check by hand:

    WRIST      = (300, 400)
    INDEX_MCP  = (240, 320)      \\
    MIDDLE_MCP = (280, 320)       |  40px apart on x, all at y=320
    RING_MCP   = (320, 320)       |
    PINKY_MCP  = (360, 320)      /

    palm_size = distance(WRIST, MIDDLE_MCP) = sqrt(20^2 + 80^2) ~= 82.46

A finger is made EXTENDED by placing its PIP directly "above" its MCP
(smaller y = higher on screen) and its TIP directly above the PIP again -
three collinear points give a straight-line angle of 180 degrees. A finger
is made CURLED by folding the TIP back down near the MCP's height, which
drops the PIP-vertex angle to under 10 degrees. Both are far past the
160-degree default threshold in either direction, so these fixtures are
robust to small threshold tuning.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GestureConfig
from modules.hand_tracker import HandData
from modules.gesture_recognition import (
    Gesture,
    FingerStates,
    GestureRecognizer,
    FingerAnalyzer,
    SwipeTracker,
    calculate_angle,
    euclidean_distance,
    get_palm_size,
    WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
)


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

_FINGER_COLUMNS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP, 240),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, 280),
    "ring": (RING_MCP, RING_PIP, RING_TIP, 320),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP, 360),
}

# Verified thumb geometry (see module docstring above for the coordinate
# scheme; angles/ratios below were computed with the exact same formulas
# this module ships, then sanity-checked against the thresholds).
_THUMB_CMC_POS = (270, 380)
_THUMB_MCP_POS = (240, 360)
_THUMB_EXTENDED = {"ip": (200, 345), "tip": (160, 330)}   # angle ~174 deg, tip far from INDEX_MCP
_THUMB_TUCKED = {"ip": (260, 355), "tip": (275, 330)}     # angle ~127 deg, tip close to INDEX_MCP but
                                                            # kept clear of any curled fingertip so it
                                                            # doesn't accidentally register as a Pinch


def build_landmarks(
    extended_fingers=(),
    thumb="tucked",
    victory_spread=False,
    thumb_tip_override=None,
):
    """
    Builds a full 21-point synthetic landmark set (pixel coordinates).

    Args:
        extended_fingers: iterable of "index"/"middle"/"ring"/"pinky" names
                           to render as EXTENDED; all others render CURLED.
        thumb: "extended" or "tucked" (see the verified constants above).
        victory_spread: when True, angles the index and middle fingertips
                         outward from their columns to widen the tip-to-tip
                         gap (used to distinguish Victory from Two Fingers -
                         see test_classify_victory_vs_two_fingers).
        thumb_tip_override: if given, overrides just the thumb TIP position
                             (used for the Pinch test, which needs the thumb
                             tip to land next to the index tip).
    """
    pts = [(0, 0)] * 21
    pts[WRIST] = (300, 400)

    pts[THUMB_CMC] = _THUMB_CMC_POS
    pts[THUMB_MCP] = _THUMB_MCP_POS
    thumb_def = _THUMB_EXTENDED if thumb == "extended" else _THUMB_TUCKED
    pts[THUMB_IP] = thumb_def["ip"]
    pts[THUMB_TIP] = thumb_tip_override if thumb_tip_override else thumb_def["tip"]

    for name, (mcp_i, pip_i, tip_i, col_x) in _FINGER_COLUMNS.items():
        pts[mcp_i] = (col_x, 320)
        pts[pip_i] = (col_x, 280)
        if name in extended_fingers:
            lateral = 0
            if victory_spread and name == "index":
                lateral = -15
            elif victory_spread and name == "middle":
                lateral = 15
            pts[tip_i] = (col_x + lateral, 220)  # collinear-ish, straight up -> ~180 deg
        else:
            pts[tip_i] = (col_x + 5, 315)  # folds back near the MCP height -> small angle

    return pts


def make_hand_data(landmarks_px, label="Right", confidence=0.95):
    landmarks_norm = [(x / 640.0, y / 480.0, 0.0) for x, y in landmarks_px]
    return HandData(
        label=label,
        confidence=confidence,
        landmarks_px=landmarks_px,
        landmarks_norm=landmarks_norm,
        bounding_box=(0, 0, 640, 480),
    )


def fresh_recognizer(**overrides) -> GestureRecognizer:
    import logging
    logger = logging.getLogger("test_gesture_recognition")
    logger.addHandler(logging.NullHandler())
    cfg = GestureConfig(**overrides)
    return GestureRecognizer(cfg, logger)


def settle(recognizer: GestureRecognizer, hand_data, frames: int = 10):
    """Feeds the same HandData repeatedly so the temporal smoothing buffer
    fills with a clear majority before we assert on the result."""
    result = None
    for _ in range(frames):
        result = recognizer.recognize(hand_data)
    return result


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def test_euclidean_distance_known_triangle():
    # Classic 3-4-5 right triangle.
    assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_euclidean_distance_zero_for_identical_points():
    assert euclidean_distance((42, 17), (42, 17)) == pytest.approx(0.0)


def test_calculate_angle_straight_line_is_180():
    # a, b, c collinear with b in the middle -> arms point in opposite
    # directions -> angle = 180 degrees.
    angle = calculate_angle((0, 10), (0, 0), (0, -10))
    assert angle == pytest.approx(180.0, abs=0.1)


def test_calculate_angle_right_angle_is_90():
    angle = calculate_angle((10, 0), (0, 0), (0, 10))
    assert angle == pytest.approx(90.0, abs=0.1)


def test_calculate_angle_acute_angle():
    # a and c both on the same side, close together -> small angle.
    angle = calculate_angle((1, 1), (0, 0), (2, 1))
    assert 0.0 <= angle < 45.0


def test_calculate_angle_degenerate_point_does_not_crash():
    # b == a (zero-length arm) must return a safe default, not raise/NaN.
    angle = calculate_angle((5, 5), (5, 5), (0, 10))
    assert angle == 180.0


def test_get_palm_size_matches_manual_calculation():
    landmarks = build_landmarks()
    expected = math.hypot(300 - 280, 400 - 320)  # WRIST -> MIDDLE_MCP
    assert get_palm_size(landmarks) == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# FingerStates
# ---------------------------------------------------------------------------

def test_finger_states_count_all_false():
    assert FingerStates().count() == 0


def test_finger_states_count_mixed():
    assert FingerStates(thumb=True, index=True, middle=False, ring=False, pinky=True).count() == 3


# ---------------------------------------------------------------------------
# FingerAnalyzer - per-finger extended/curled classification
# ---------------------------------------------------------------------------

def test_finger_analyzer_all_fingers_extended():
    landmarks = build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended")
    analyzer = FingerAnalyzer(GestureConfig())
    states, confidences = analyzer.analyze(landmarks)
    assert states.as_tuple() == (True, True, True, True, True)
    # A cleanly straight/bent fixture should read with high confidence.
    # (An angle of exactly 180 degrees maps to exactly 0.5 confidence given
    # the 40-degree normalization span in _angle_to_confidence, so we check
    # >= rather than a strict >.)
    assert all(c >= 0.5 for c in confidences.values())


def test_finger_analyzer_all_fingers_curled():
    landmarks = build_landmarks(extended_fingers=[], thumb="tucked")
    analyzer = FingerAnalyzer(GestureConfig())
    states, _ = analyzer.analyze(landmarks)
    assert states.as_tuple() == (False, False, False, False, False)


def test_finger_analyzer_thumb_extended_vs_tucked_independent_of_other_fingers():
    analyzer = FingerAnalyzer(GestureConfig())

    extended_landmarks = build_landmarks(extended_fingers=[], thumb="extended")
    tucked_landmarks = build_landmarks(extended_fingers=[], thumb="tucked")

    states_ext, _ = analyzer.analyze(extended_landmarks)
    states_tuck, _ = analyzer.analyze(tucked_landmarks)

    assert states_ext.thumb is True
    assert states_tuck.thumb is False


# ---------------------------------------------------------------------------
# GestureRecognizer - static pose classification (via the full recognizer,
# so temporal smoothing and confidence blending are exercised too)
# ---------------------------------------------------------------------------

def test_classify_closed_fist():
    recognizer = fresh_recognizer()
    landmarks = build_landmarks(extended_fingers=[], thumb="tucked")
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.CLOSED_FIST
    assert result.finger_count == 0
    assert result.is_tracking is True


def test_classify_open_palm():
    recognizer = fresh_recognizer()
    landmarks = build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended")
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.OPEN_PALM
    assert result.finger_count == 5


def test_classify_index_finger():
    recognizer = fresh_recognizer()
    landmarks = build_landmarks(extended_fingers=["index"], thumb="tucked")
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.INDEX_FINGER


def test_classify_two_fingers_when_tips_are_close_together():
    recognizer = fresh_recognizer()
    # Straight-up tips on their default columns: 40px apart, well under
    # the 0.5 * palm_size (~41px) spread threshold -> "together".
    landmarks = build_landmarks(extended_fingers=["index", "middle"], thumb="tucked", victory_spread=False)
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.TWO_FINGERS


def test_classify_victory_when_tips_are_spread_apart():
    recognizer = fresh_recognizer()
    # Same finger states as the "together" case, but angled outward so
    # the tip-to-tip gap clears the spread threshold.
    landmarks = build_landmarks(extended_fingers=["index", "middle"], thumb="tucked", victory_spread=True)
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.VICTORY


def test_classify_thumbs_up():
    recognizer = fresh_recognizer()
    landmarks = build_landmarks(extended_fingers=[], thumb="extended")
    # Point the extended thumb tip upward (smaller y) relative to the wrist.
    landmarks[THUMB_TIP] = (240, 250)
    landmarks[THUMB_IP] = (240, 300)
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.THUMBS_UP


def test_classify_pinch_overrides_other_finger_states():
    recognizer = fresh_recognizer()
    landmarks = build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended")
    # Move the thumb tip right next to the (extended) index tip.
    landmarks[THUMB_TIP] = (245, 225)
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.PINCH


def test_classify_finger_count_fallback_for_unnamed_combination():
    recognizer = fresh_recognizer()
    # Middle + ring extended, index + pinky curled: not one of the named
    # poses, so this should fall back to the generic FINGER_COUNT label.
    landmarks = build_landmarks(extended_fingers=["middle", "ring"], thumb="tucked")
    result = settle(recognizer, make_hand_data(landmarks))
    assert result.gesture == Gesture.FINGER_COUNT
    assert result.finger_count == 2


# ---------------------------------------------------------------------------
# Temporal smoothing / noise rejection
# ---------------------------------------------------------------------------

def test_smoothing_confirms_gesture_once_majority_reached():
    recognizer = fresh_recognizer(smoothing_window=7, smoothing_majority_ratio=0.6)
    landmarks = build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended")
    hand_data = make_hand_data(landmarks)
    # Fewer than a majority of frames seen so far -> should not yet be
    # confidently OPEN_PALM (still DETECTING) on the very first frame.
    first_result = recognizer.recognize(hand_data)
    assert first_result.gesture in (Gesture.DETECTING, Gesture.OPEN_PALM)
    # After enough repeated identical frames, the majority vote must confirm it.
    for _ in range(9):
        result = recognizer.recognize(hand_data)
    assert result.gesture == Gesture.OPEN_PALM


def test_smoothing_rejects_a_single_noisy_frame():
    recognizer = fresh_recognizer(smoothing_window=7, smoothing_majority_ratio=0.6)
    palm = make_hand_data(build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended"))
    fist = make_hand_data(build_landmarks(extended_fingers=[], thumb="tucked"))

    for _ in range(7):
        result = recognizer.recognize(palm)
    assert result.gesture == Gesture.OPEN_PALM

    # One single noisy/misread frame should not be enough to flip the
    # smoothed output away from the established majority.
    result = recognizer.recognize(fist)
    assert result.gesture == Gesture.OPEN_PALM


def test_smoothing_reports_detecting_when_no_clear_majority():
    # 7-frame window, 0.6 majority ratio -> need >= 5 agreeing frames
    # (4/7 = 0.57 falls just short). An alternating A/B/A/B/... sequence
    # gives the majority label only 4 of 7 votes, which must NOT confirm.
    recognizer = fresh_recognizer(smoothing_window=7, smoothing_majority_ratio=0.6)
    palm = make_hand_data(build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended"))
    fist = make_hand_data(build_landmarks(extended_fingers=[], thumb="tucked"))

    sequence = [palm, fist, palm, fist, palm, fist, palm]  # palm=4, fist=3 out of 7
    result = None
    for hand_data in sequence:
        result = recognizer.recognize(hand_data)
    assert result.gesture == Gesture.DETECTING


# ---------------------------------------------------------------------------
# No-hand / tracking-lost behaviour
# ---------------------------------------------------------------------------

def test_recognize_with_no_hand_returns_none_and_not_tracking():
    recognizer = fresh_recognizer()
    result = recognizer.recognize(None)
    assert result.gesture == Gesture.NONE
    assert result.is_tracking is False
    assert result.confidence == 0.0
    assert result.finger_states is None
    assert result.finger_count == 0


def test_losing_and_reacquiring_hand_does_not_carry_over_smoothing_history():
    recognizer = fresh_recognizer(smoothing_window=7, smoothing_majority_ratio=0.6)
    palm = make_hand_data(build_landmarks(extended_fingers=["index", "middle", "ring", "pinky"], thumb="extended"))
    for _ in range(7):
        recognizer.recognize(palm)

    recognizer.recognize(None)  # hand lost -> history/swipe tracker must reset
    assert len(recognizer._history) == 0

    # A single frame right after reacquiring should not instantly inherit
    # the old majority; it needs to build a new majority from scratch.
    fist = make_hand_data(build_landmarks(extended_fingers=[], thumb="tucked"))
    result = recognizer.recognize(fist)
    assert result.gesture in (Gesture.DETECTING, Gesture.CLOSED_FIST)


# ---------------------------------------------------------------------------
# SwipeTracker
# ---------------------------------------------------------------------------

def _feed_linear_motion(tracker: SwipeTracker, start, end, steps, palm_size=80.0):
    """Feeds `steps` evenly spaced positions from start to end; returns the
    last non-None result, or None if no swipe fired anywhere in the path."""
    last_result = None
    for i in range(steps):
        t = i / (steps - 1)
        pos = (
            int(start[0] + (end[0] - start[0]) * t),
            int(start[1] + (end[1] - start[1]) * t),
        )
        result = tracker.update(pos, palm_size)
        if result is not None:
            last_result = result
    return last_result


def test_swipe_tracker_detects_swipe_right():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2))
    # dx = 150 over palm_size 80 -> norm_dx ~= 1.875, well past the 1.2 threshold.
    result = _feed_linear_motion(tracker, (200, 300), (350, 300), steps=10)
    assert result == Gesture.SWIPE_RIGHT


def test_swipe_tracker_detects_swipe_left():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2))
    result = _feed_linear_motion(tracker, (350, 300), (200, 300), steps=10)
    assert result == Gesture.SWIPE_LEFT


def test_swipe_tracker_detects_swipe_up():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2))
    # Screen y decreases upward, so "up" is a negative dy.
    result = _feed_linear_motion(tracker, (300, 350), (300, 200), steps=10)
    assert result == Gesture.SWIPE_UP


def test_swipe_tracker_detects_swipe_down():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2))
    result = _feed_linear_motion(tracker, (300, 200), (300, 350), steps=10)
    assert result == Gesture.SWIPE_DOWN


def test_swipe_tracker_ignores_small_motion():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2))
    # dx = 20 over palm_size 80 -> norm_dx = 0.25, well under the threshold.
    result = _feed_linear_motion(tracker, (300, 300), (320, 300), steps=10)
    assert result is None


def test_swipe_tracker_ignores_diagonal_motion():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2))
    # Equal dx/dy: large enough magnitude, but neither axis dominates.
    result = _feed_linear_motion(tracker, (250, 250), (350, 350), steps=10)
    assert result is None


def test_swipe_tracker_cooldown_prevents_immediate_retrigger():
    tracker = SwipeTracker(
        GestureConfig(swipe_history_len=10, swipe_min_distance_ratio=1.2, swipe_cooldown_frames=15)
    )
    first = _feed_linear_motion(tracker, (200, 300), (350, 300), steps=10)
    assert first == Gesture.SWIPE_RIGHT

    # Immediately repeating the same motion should be suppressed by the
    # cooldown counter rather than firing a second SWIPE_RIGHT right away.
    second = _feed_linear_motion(tracker, (200, 300), (350, 300), steps=10)
    assert second is None


def test_swipe_tracker_reset_clears_history():
    tracker = SwipeTracker(GestureConfig(swipe_history_len=10))
    tracker.update((300, 300), 80.0)
    tracker.update((310, 300), 80.0)
    tracker.reset()
    assert len(tracker._positions) == 0
    assert tracker._cooldown == 0
