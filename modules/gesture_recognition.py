"""
modules/gesture_recognition.py
================================
Phase 2 of the AR-Based Gesture Controlled System: turns the raw 21-point
hand landmarks produced by Phase 1's `HandTracker` into a named, smoothed,
confidence-scored gesture.

This module does NOT modify anything in Phase 1. It only CONSUMES the
`HandData` dataclass that `modules/hand_tracker.py` already returns, exactly
as that module's own docstring predicted it would: "Phase 2 (gesture
classification) will consume the HandData objects this module returns
without needing to touch MediaPipe directly at all."

--------------------------------------------------------------------------
THE MATHEMATICS OF GESTURE RECOGNITION (read this before the code below)
--------------------------------------------------------------------------

A "gesture" here is really just a label we attach to a specific geometric
configuration of 21 points in space. There are three families of geometry
this module uses, all built from nothing more than vector arithmetic:

1) FINGER ANGLES  (are_they_bent / are_they_straight)
   -----------------------------------------------------
   Every non-thumb finger has four landmarks in a chain: MCP -> PIP -> TIP
   (we use MCP, PIP and TIP; skipping DIP keeps the rule simple without
   losing meaningful accuracy). Picture the PIP joint as the vertex of an
   angle whose two arms point toward the MCP and the TIP:

       MCP                                   MCP
        \\                                     \\
         \\  angle ~ 180 deg (straight)         *---TIP   angle ~ 60-90 deg
          PIP-------TIP                        /             (curled)
                                              PIP

   We compute this angle with the standard dot-product formula. If
   v1 = MCP - PIP and v2 = TIP - PIP, then the angle between them is:

       theta = arccos( (v1 . v2) / (|v1| * |v2|) )

   A straight finger points v1 and v2 in nearly OPPOSITE directions (the
   MCP and TIP sit on either side of the PIP in a straight line), so their
   dot product is strongly negative and theta approaches 180 degrees. A
   curled finger folds the TIP back toward the MCP side, shrinking theta
   toward 90 degrees or less. We call a finger EXTENDED when
   theta > finger_angle_threshold (default 160 degrees).

   The thumb does not have an MCP/PIP/TIP chain that bends the same way
   (its joints rotate roughly perpendicular to the other fingers'), so it
   gets its own rule combining an angle check at its IP joint with a
   distance check (see below) - this redundancy is deliberate; see
   "Reducing False Positives" further down.

2) DISTANCES (how far apart are two points, in a scale-invariant way)
   ---------------------------------------------------------------------
   Raw pixel distance is useless on its own: a hand held close to the
   camera produces huge pixel distances, a hand held far away produces
   tiny ones, for the exact same physical gesture. Every distance in this
   module is therefore expressed as a RATIO against the hand's own
   "palm size" - the distance between the WRIST and the MIDDLE_MCP
   landmarks, which stays roughly proportional to the hand's apparent
   size at any distance from the camera:

       palm_size = euclidean_distance(WRIST, MIDDLE_FINGER_MCP)
       ratio     = euclidean_distance(point_a, point_b) / palm_size

   This is what makes a PINCH threshold (ratio < 0.4) work whether the
   user's hand fills the frame or sits far in the background.

3) RELATIVE POSITIONS (where is a point compared to another, along one axis)
   ---------------------------------------------------------------------------
   Some gestures are less about a joint angle and more about *where*
   something sits. "Thumbs Up" isn't just "thumb extended" - it's "thumb
   extended AND pointing up". Since OpenCV image coordinates increase
   DOWNWARD (y=0 is the top of the frame), "pointing up on screen" means
   a SMALLER y value than the reference point:

       thumb_tip.y < wrist.y - (margin * palm_size)   ->  thumb points up

4) TEMPORAL TRACKING (swipes, and smoothing every other gesture too)
   ---------------------------------------------------------------------
   A single frame is a snapshot; a swipe is a *motion*, so it cannot be
   read from one frame at all. We keep a short rolling history (a
   `collections.deque`) of the wrist's pixel position over the last
   `swipe_history_len` frames. Each new frame we compute the net
   displacement between the oldest and newest point in that window,
   normalize it by the average palm size over the window (same
   scale-invariance idea as above), and check two things:
     - was the total displacement large enough to be deliberate
       (> swipe_min_distance_ratio times the hand's own size)?
     - was it clearly dominated by one axis (mostly horizontal or
       mostly vertical, not diagonal)?
   If both hold, we emit the matching SWIPE_* gesture and then apply a
   short cooldown (ignore new swipes for a few frames) so one physical
   swipe doesn't get reported five times as the hand decelerates.

   Every OTHER gesture (poses, not motions) is also smoothed temporally:
   we keep a short history of per-frame raw classifications and only
   "confirm" a gesture once it has a clear majority in that window. This
   is a simple, cheap, and very effective noise filter - see below.

5) CONFIDENCE SCORE
   ---------------------------------------------------------------------
   Final confidence blends two independent signals:
     geometric_confidence  - how far past its threshold each relevant
                              angle/distance check landed on THIS frame
                              (a finger at 179 degrees is a far more
                              confident "extended" than one at 161).
     temporal_confidence   - what fraction of the recent frame-history
                              buffer agrees with the label we're about
                              to report.
     final = 0.5 * geometric_confidence + 0.5 * temporal_confidence

6) REDUCING FALSE POSITIVES
   ---------------------------------------------------------------------
   Rule-based classifiers are cheap and interpretable but fragile at the
   boundary between two classes. This module fights that in five ways:
     - Scale-invariant ratios (section 2) instead of raw pixel numbers.
     - Redundant checks for the trickiest finger (the thumb gets BOTH an
       angle test and a distance test; both must agree).
     - Majority-vote temporal smoothing (section 4) so a single
       misread frame - caused by motion blur, brief occlusion, or a
       MediaPipe landmark jitter - cannot flip the displayed gesture.
     - A minimum-confidence floor (`min_confidence_to_display`): if
       nothing clears it, we report `Gesture.DETECTING` rather than a
       possibly-wrong guess.
     - A swipe cooldown window so a single physical motion cannot be
       re-reported multiple times as multiple separate swipes.

--------------------------------------------------------------------------
HOW THIS WOULD LATER BE REPLACED WITH MACHINE LEARNING
--------------------------------------------------------------------------
Every rule above is really just a hand-written decision boundary over a
feature vector: 5 finger angles, a handful of normalized distances, and a
couple of relative-position checks. That feature vector is *exactly* the
kind of input a small classifier (scikit-learn RandomForestClassifier /
SVM, or a tiny feed-forward network) is trained on. The migration path is
deliberately a single seam:

    1. Log `FingerAnalyzer.analyze()`'s output (or the raw 21x3 landmark
       array) alongside a human-provided label for many recorded gestures.
    2. Train a classifier offline on that dataset.
    3. Replace the body of `GestureRecognizer._classify_static_pose()`
       with `self._model.predict(feature_vector)`, leaving everything
       else - HandData input, palm-size normalization, temporal
       smoothing, swipe tracking, confidence blending, the on-screen
       overlay - completely untouched.

The stub method `GestureRecognizer._classify_with_ml()` below marks
exactly where that swap would happen; it is intentionally unimplemented
in Phase 2, which asks for the rule-based version first.
"""

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config import GestureConfig
from modules.hand_tracker import HandData
from modules.utils import draw_text_with_background


# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------
# These mirror HandTracker.LANDMARK_NAMES (Phase 1) by position. Defining
# them by name here - rather than importing magic numbers - is what makes
# the geometry functions below readable ("landmarks[INDEX_TIP]" instead of
# "landmarks[8]").
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class Gesture(Enum):
    """Every gesture label this engine can emit, plus two transient states."""
    NONE = "None"                    # no hand currently tracked
    DETECTING = "Detecting..."       # a hand is tracked but confidence/majority not yet reached
    OPEN_PALM = "Open Palm"
    CLOSED_FIST = "Closed Fist"
    INDEX_FINGER = "Index Finger"
    TWO_FINGERS = "Two Fingers"
    VICTORY = "Victory"
    THUMBS_UP = "Thumbs Up"
    PINCH = "Pinch"
    FINGER_COUNT = "Finger Count"     # fallback: reports FingerStates.count()
    SWIPE_LEFT = "Swipe Left"
    SWIPE_RIGHT = "Swipe Right"
    SWIPE_UP = "Swipe Up"
    SWIPE_DOWN = "Swipe Down"


_SWIPE_GESTURES = {
    Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT, Gesture.SWIPE_UP, Gesture.SWIPE_DOWN,
}


@dataclass
class FingerStates:
    """Boolean extended/curled state for each of the 5 fingers on one hand."""
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    def count(self) -> int:
        """Number of fingers currently extended (0-5). Used for FINGER_COUNT."""
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])

    def as_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


@dataclass
class GestureResult:
    """
    Everything a caller (app.py, or any future consumer) needs for one frame.

    `raw_gesture` is the pre-smoothing, single-frame classification; useful
    for debugging/logging even though `gesture` (the smoothed, confirmed
    label) is what should actually drive any downstream action.
    """
    gesture: Gesture
    confidence: float                 # 0.0 - 1.0, see module docstring section 5
    finger_states: Optional[FingerStates]
    finger_count: int
    is_tracking: bool                 # False when no hand is currently visible
    raw_gesture: Gesture               # instantaneous, unsmoothed classification


# ---------------------------------------------------------------------------
# Pure math helpers - free functions, independent of any class, so they are
# trivial to unit test in isolation (see tests/test_gesture_recognition.py).
# ---------------------------------------------------------------------------

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Straight-line distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Angle ABC in degrees: the angle at vertex `b`, between the arms b->a
    and b->c. Implements theta = arccos( (v1 . v2) / (|v1| |v2|) ) from
    the module docstring's "Finger Angles" section.

    Returns 180.0 for degenerate (zero-length) arms rather than raising,
    so a momentarily-invalid landmark can't crash the whole pipeline -
    it will simply read as "straight", and the temporal smoothing /
    confidence-floor machinery absorbs the occasional bad frame.
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)  # guards against float rounding pushing |cosine| slightly past 1
    return float(np.degrees(np.arccos(cosine)))


def get_palm_size(landmarks_px: List[Tuple[int, int]]) -> float:
    """
    Scale reference used to normalize every distance in this module:
    WRIST-to-MIDDLE_MCP distance, which grows/shrinks with how close the
    hand is to the camera in roughly the same proportion as everything
    else on the hand.
    """
    size = euclidean_distance(landmarks_px[WRIST], landmarks_px[MIDDLE_MCP])
    return max(size, 1e-6)  # avoid a division-by-zero if landmarks are ever degenerate


# ---------------------------------------------------------------------------
# Finger state analysis
# ---------------------------------------------------------------------------

class FingerAnalyzer:
    """
    Converts 21 raw landmarks into a `FingerStates` boolean vector, plus a
    per-finger confidence dict describing how far past its threshold each
    finger's reading landed (used for the geometric half of the final
    confidence score - see module docstring section 5).
    """

    # (MCP, PIP, TIP) index triples for the four fingers that share the
    # same joint geometry. The thumb is handled separately in `_thumb_state`.
    _FINGER_JOINTS = {
        "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        "ring": (RING_MCP, RING_PIP, RING_TIP),
        "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    }

    def __init__(self, config: GestureConfig):
        self.config = config

    def analyze(
        self, landmarks_px: List[Tuple[int, int]]
    ) -> Tuple[FingerStates, Dict[str, float]]:
        """
        Returns:
            states:      FingerStates with one bool per finger.
            confidences: {"thumb": 0.83, "index": 0.41, ...} - each value
                         is how far past (or short of) the extended
                         threshold that finger's angle landed, normalized
                         to roughly [0, 1]. Used to compute geometric
                         confidence for whichever gesture ends up matching.
        """
        confidences: Dict[str, float] = {}
        states_kwargs: Dict[str, bool] = {}

        for name, (mcp, pip, tip) in self._FINGER_JOINTS.items():
            angle = calculate_angle(landmarks_px[mcp], landmarks_px[pip], landmarks_px[tip])
            extended = angle > self.config.finger_angle_threshold
            states_kwargs[name] = extended
            confidences[name] = self._angle_to_confidence(angle, extended)

        thumb_extended, thumb_confidence = self._thumb_state(landmarks_px)
        states_kwargs["thumb"] = thumb_extended
        confidences["thumb"] = thumb_confidence

        return FingerStates(**states_kwargs), confidences

    def _angle_to_confidence(self, angle: float, extended: bool) -> float:
        """
        Maps "how far past the threshold" to roughly [0, 1]. The 40-degree
        span below is just a normalization constant (a finger fully
        straight at 180 deg or fully curled at 90 deg or below is treated
        as maximally confident); it is not itself a decision threshold.
        """
        threshold = self.config.finger_angle_threshold
        if extended:
            return float(np.clip((angle - threshold) / 40.0, 0.0, 1.0))
        return float(np.clip((threshold - angle) / 40.0, 0.0, 1.0))

    def _thumb_state(self, landmarks_px: List[Tuple[int, int]]) -> Tuple[bool, float]:
        """
        The thumb's joints rotate roughly perpendicular to the other
        fingers', so a single MCP/PIP/TIP angle check (as used for the
        other four fingers) is unreliable on its own. We combine TWO
        independent signals and require both to agree - the redundant
        cross-check mentioned in the module docstring's false-positives
        section:
          1. Angle at the thumb's IP joint (CMC->IP arm vs TIP->IP arm).
          2. Distance from the thumb tip to the INDEX finger's MCP,
             normalized by palm size. A curled thumb tucks in across the
             front of the palm toward the index/middle side (the radial
             side of the hand), landing close to the index MCP; an
             abducted, extended thumb swings away from it. (The pinky's
             MCP sits on the far, ulnar side of the palm and stays a
             similar distance from the thumb tip in both states, so it
             is a poor reference point for this check.)
        """
        angle = calculate_angle(landmarks_px[THUMB_CMC], landmarks_px[THUMB_IP], landmarks_px[THUMB_TIP])
        angle_extended = angle > self.config.thumb_angle_threshold

        palm_size = get_palm_size(landmarks_px)
        tuck_ratio = euclidean_distance(landmarks_px[THUMB_TIP], landmarks_px[INDEX_MCP]) / palm_size
        distance_extended = tuck_ratio > self.config.thumb_tuck_ratio

        extended = angle_extended and distance_extended

        angle_conf = np.clip((angle - self.config.thumb_angle_threshold) / 40.0, 0.0, 1.0) if angle_extended \
            else np.clip((self.config.thumb_angle_threshold - angle) / 40.0, 0.0, 1.0)
        dist_conf = np.clip((tuck_ratio - self.config.thumb_tuck_ratio) / 0.3, 0.0, 1.0) if distance_extended \
            else np.clip((self.config.thumb_tuck_ratio - tuck_ratio) / 0.3, 0.0, 1.0)

        return extended, float((angle_conf + dist_conf) / 2.0)


# ---------------------------------------------------------------------------
# Swipe (motion) tracking
# ---------------------------------------------------------------------------

class SwipeTracker:
    """
    Detects directional swipes from the WRIST landmark's motion over a
    short rolling window. See module docstring section 4 for the algorithm.
    """

    def __init__(self, config: GestureConfig):
        self.config = config
        self._positions: Deque[Tuple[int, int]] = deque(maxlen=config.swipe_history_len)
        self._palm_sizes: Deque[float] = deque(maxlen=config.swipe_history_len)
        self._cooldown = 0

    def reset(self) -> None:
        """Clears history - called whenever hand tracking is lost, so a
        swipe is never synthesized from two unrelated tracking sessions."""
        self._positions.clear()
        self._palm_sizes.clear()
        self._cooldown = 0

    def update(self, wrist_px: Tuple[int, int], palm_size: float) -> Optional[Gesture]:
        """
        Call once per tracked frame. Returns a SWIPE_* Gesture the moment
        a deliberate, axis-dominant motion is detected, else None.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            # Still record position during cooldown so the window stays
            # populated with recent motion once the cooldown lapses.
            self._positions.append(wrist_px)
            self._palm_sizes.append(palm_size)
            return None

        self._positions.append(wrist_px)
        self._palm_sizes.append(palm_size)

        if len(self._positions) < self._positions.maxlen:
            return None  # not enough history yet to judge a swipe

        oldest = self._positions[0]
        newest = self._positions[-1]
        dx = newest[0] - oldest[0]
        dy = newest[1] - oldest[1]

        avg_palm_size = float(np.mean(self._palm_sizes))
        norm_dx = dx / avg_palm_size
        norm_dy = dy / avg_palm_size
        magnitude = math.hypot(norm_dx, norm_dy)

        if magnitude < self.config.swipe_min_distance_ratio:
            return None  # motion too small to be a deliberate swipe

        # Require the motion to be clearly dominated by one axis so a
        # diagonal hand movement doesn't get misreported as left/right.
        if abs(norm_dx) > 2 * abs(norm_dy):
            gesture = Gesture.SWIPE_RIGHT if norm_dx > 0 else Gesture.SWIPE_LEFT
        elif abs(norm_dy) > 2 * abs(norm_dx):
            gesture = Gesture.SWIPE_DOWN if norm_dy > 0 else Gesture.SWIPE_UP
        else:
            return None  # too diagonal to confidently call a direction

        # Start the cooldown and clear history so this single physical
        # swipe cannot immediately re-trigger on the very next frame.
        self._cooldown = self.config.swipe_cooldown_frames
        self._positions.clear()
        self._palm_sizes.clear()
        return gesture


# ---------------------------------------------------------------------------
# The recognizer itself
# ---------------------------------------------------------------------------

class GestureRecognizer:
    """
    High-level, per-frame entry point. Typical usage (mirrors HandTracker's
    own usage pattern from Phase 1):

        recognizer = GestureRecognizer(config.gesture, logger)
        ...
        primary_hand = hands_data[0] if hands_data else None
        result = recognizer.recognize(primary_hand)
        recognizer.draw_overlay(annotated_frame, result)
    """

    def __init__(self, gesture_config: GestureConfig, logger: logging.Logger):
        self.config = gesture_config
        self.logger = logger
        self._finger_analyzer = FingerAnalyzer(gesture_config)
        self._swipe_tracker = SwipeTracker(gesture_config)
        self._history: Deque[Gesture] = deque(maxlen=gesture_config.smoothing_window)

        self.logger.info(
            "GestureRecognizer initialized (angle_threshold=%.1f deg, "
            "smoothing_window=%d, min_confidence=%.2f)",
            gesture_config.finger_angle_threshold,
            gesture_config.smoothing_window,
            gesture_config.min_confidence_to_display,
        )

    def recognize(self, hand_data: Optional[HandData]) -> GestureResult:
        """
        Runs the full Phase 2 pipeline for one frame: finger analysis ->
        static-pose classification -> swipe check -> temporal smoothing ->
        confidence blending. Safe to call with `hand_data=None` (no hand
        currently tracked); this resets the swipe tracker so a lost-and-
        reacquired hand never produces a spurious swipe from the gap.
        """
        if hand_data is None:
            self._swipe_tracker.reset()
            self._history.clear()
            return GestureResult(
                gesture=Gesture.NONE,
                confidence=0.0,
                finger_states=None,
                finger_count=0,
                is_tracking=False,
                raw_gesture=Gesture.NONE,
            )

        landmarks_px = hand_data.landmarks_px
        finger_states, per_finger_confidence = self._finger_analyzer.analyze(landmarks_px)
        palm_size = get_palm_size(landmarks_px)

        raw_gesture, geometric_confidence = self._classify_static_pose(
            finger_states, landmarks_px, palm_size, per_finger_confidence
        )

        # A swipe is a motion event: when one fires, it overrides whatever
        # static pose was read on this same frame, since a swipe is a much
        # more deliberate, high-confidence signal than a passive pose.
        swipe = self._swipe_tracker.update(landmarks_px[WRIST], palm_size)
        if swipe is not None:
            raw_gesture = swipe
            geometric_confidence = 1.0

        self._history.append(raw_gesture)
        smoothed_gesture, temporal_confidence = self._smooth()

        final_confidence = float(0.5 * geometric_confidence + 0.5 * temporal_confidence)

        # A swipe is transient by nature - report it for exactly the one
        # frame it fires on rather than waiting for it to win a majority
        # vote it will never get again (the hand keeps moving afterward).
        if raw_gesture in _SWIPE_GESTURES:
            smoothed_gesture = raw_gesture
            final_confidence = 1.0

        if smoothed_gesture != Gesture.NONE and final_confidence < self.config.min_confidence_to_display:
            smoothed_gesture = Gesture.DETECTING

        return GestureResult(
            gesture=smoothed_gesture,
            confidence=final_confidence,
            finger_states=finger_states,
            finger_count=finger_states.count(),
            is_tracking=True,
            raw_gesture=raw_gesture,
        )

    def _classify_static_pose(
        self,
        fingers: FingerStates,
        landmarks_px: List[Tuple[int, int]],
        palm_size: float,
        per_finger_confidence: Dict[str, float],
    ) -> Tuple[Gesture, float]:
        """
        Priority-ordered rule chain. Order matters: more specific / more
        deliberate gestures are checked before generic fallbacks, so e.g.
        a Pinch is never accidentally reported as a 4-finger "Finger
        Count" just because the untucked fingers happen to be extended.
        """
        thumb_index_ratio = euclidean_distance(
            landmarks_px[THUMB_TIP], landmarks_px[INDEX_TIP]
        ) / palm_size

        # 1) Pinch - checked first; a deliberate thumb-index touch should
        #    win regardless of what the other fingers are doing.
        if thumb_index_ratio < self.config.pinch_distance_ratio:
            conf = np.clip((self.config.pinch_distance_ratio - thumb_index_ratio) / self.config.pinch_distance_ratio, 0.0, 1.0)
            return Gesture.PINCH, float(conf)

        # 2) Closed Fist - every finger, including the thumb, curled.
        if fingers.count() == 0:
            conf = np.mean(list(per_finger_confidence.values()))
            return Gesture.CLOSED_FIST, float(conf)

        # 3) Thumbs Up - only the thumb extended, AND it points upward
        #    on screen (smaller y = higher up, per module docstring #3).
        if fingers.thumb and not (fingers.index or fingers.middle or fingers.ring or fingers.pinky):
            thumb_tip_y = landmarks_px[THUMB_TIP][1]
            wrist_y = landmarks_px[WRIST][1]
            pointing_up = thumb_tip_y < wrist_y - (0.2 * palm_size)
            if pointing_up:
                return Gesture.THUMBS_UP, per_finger_confidence["thumb"]

        # 4) Open Palm - all five fingers extended.
        if fingers.count() == 5:
            conf = np.mean(list(per_finger_confidence.values()))
            return Gesture.OPEN_PALM, float(conf)

        # 5) Victory vs 6) Two Fingers - both are "index + middle extended,
        #    ring + pinky curled"; distinguished only by how far apart the
        #    two tips are spread, normalized by palm size.
        if fingers.index and fingers.middle and not fingers.ring and not fingers.pinky:
            spread_ratio = euclidean_distance(
                landmarks_px[INDEX_TIP], landmarks_px[MIDDLE_TIP]
            ) / palm_size
            conf = float(np.mean([per_finger_confidence["index"], per_finger_confidence["middle"]]))
            if spread_ratio > self.config.spread_distance_ratio:
                return Gesture.VICTORY, conf
            return Gesture.TWO_FINGERS, conf

        # 7) Index Finger - only the index finger extended among the four.
        if fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky:
            return Gesture.INDEX_FINGER, per_finger_confidence["index"]

        # 8) Fallback - no named pose matched; report the raw count
        #    (covers e.g. middle+ring extended, or exactly 3 fingers up).
        conf = np.mean(list(per_finger_confidence.values()))
        return Gesture.FINGER_COUNT, float(conf)

    def _smooth(self) -> Tuple[Gesture, float]:
        """
        Majority-vote temporal smoothing over `self._history` (see module
        docstring section 4). Returns (majority_label, agreement_ratio).
        An empty or not-yet-full history simply votes on whatever it has.
        """
        if not self._history:
            return Gesture.NONE, 0.0

        counts = Counter(self._history)
        majority_gesture, majority_count = counts.most_common(1)[0]
        agreement_ratio = majority_count / len(self._history)

        if agreement_ratio < self.config.smoothing_majority_ratio:
            return Gesture.DETECTING, agreement_ratio

        return majority_gesture, agreement_ratio

    def _classify_with_ml(self, feature_vector: np.ndarray):
        """
        Not implemented in Phase 2 - see the module docstring's "How This
        Would Later Be Replaced With Machine Learning" section for the
        exact migration path. This stub exists purely to mark, in code,
        where that swap would happen: a trained `self._model.predict(...)`
        call would replace `_classify_static_pose()`'s rule chain, while
        every other part of `recognize()` (temporal smoothing, swipe
        tracking, confidence blending, the on-screen overlay) stays as-is.
        """
        raise NotImplementedError(
            "Phase 2 uses rule-based classification only. This hook is "
            "reserved for a future ML-backed classifier trained on the "
            "same feature vector FingerAnalyzer already computes."
        )

    def draw_overlay(self, frame: np.ndarray, result: GestureResult, origin: Tuple[int, int] = (10, 100)) -> None:
        """
        Draws the Phase 2 status block (detected gesture, confidence,
        tracking status) onto `frame` in place, directly below Phase 1's
        existing FPS / hand-count overlay drawn by app.py. Reuses Phase
        1's own `draw_text_with_background` helper rather than duplicating
        drawing code.
        """
        x, y = origin
        line_height = 30

        tracking_text = "Tracking: Active" if result.is_tracking else "Tracking: Lost"
        draw_text_with_background(frame, tracking_text, (x, y))

        gesture_text = f"Gesture: {result.gesture.value}"
        draw_text_with_background(frame, gesture_text, (x, y + line_height))

        confidence_text = f"Confidence: {result.confidence * 100:.0f}%"
        draw_text_with_background(frame, confidence_text, (x, y + 2 * line_height))
