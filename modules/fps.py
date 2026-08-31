"""
modules/fps.py
===============
FPS (Frames Per Second) measurement utility.

Why a dedicated module?
- FPS logic is needed for on-screen display NOW, and will be needed for
  performance logging/analytics dashboards in later phases. Isolating it
  here avoids duplicating timing code inside app.py or hand_tracker.py.

Approach:
- Uses an Exponential Moving Average (EMA) instead of a raw frame-to-frame
  delta, so the displayed number doesn't jitter wildly between frames
  (e.g. 28 -> 45 -> 22 -> 40). This gives a stable, readable FPS counter.
"""

import time


class FPS:
    """Tracks and smooths real-time frames-per-second using an EMA filter."""

    def __init__(self, smoothing: float = 0.9):
        """
        Args:
            smoothing: weight (0-1) given to the PREVIOUS FPS estimate.
                       Higher = smoother display but slower to react to
                       real performance changes. 0.9 is a good default
                       for a live camera preview.
        """
        self._prev_time = time.time()
        self._fps = 0.0
        self._smoothing = smoothing

    def update(self) -> float:
        """
        Call exactly once per processed frame (i.e. once per loop
        iteration in app.py, after all processing for that frame is done).

        Returns:
            The current smoothed FPS value.
        """
        current_time = time.time()
        elapsed = current_time - self._prev_time
        self._prev_time = current_time

        if elapsed <= 0:
            # Guards against a divide-by-zero on the very first call or
            # on systems with low timer resolution.
            return self._fps

        instantaneous_fps = 1.0 / elapsed

        # EMA formula: new_estimate = alpha * old_estimate + (1 - alpha) * new_sample
        self._fps = (self._smoothing * self._fps) + ((1 - self._smoothing) * instantaneous_fps)
        return self._fps

    def get(self) -> float:
        """Returns the last computed FPS value without advancing the timer."""
        return self._fps
