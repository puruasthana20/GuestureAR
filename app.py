"""
app.py
=======
Entry point for Phase 1 of the AR-Based Gesture Controlled System.

Responsibilities:
  1. Initialize logging + load config.
  2. Open the webcam via the Camera module (as a context manager, so it
     is guaranteed to be released even if something crashes).
  3. Run every captured frame through HandTracker to get landmarks +
     an annotated preview frame.
  4. Overlay FPS and hand-count text.
  5. Handle keyboard controls (quit, screenshot, toggle overlays).
  6. Guarantee clean shutdown of camera + MediaPipe resources on exit,
     whether that exit is normal (user pressed 'q') or an exception.

Run with:   python app.py
Quit with:  'q' key, or close the OpenCV window
"""

import sys
import cv2

from config import config
from modules.utils import setup_logging, save_screenshot, draw_text_with_background
from modules.camera import Camera, CameraError
from modules.hand_tracker import HandTracker
from modules.fps import FPS
from modules.gesture_recognition import GestureRecognizer  # Phase 2 addition


def print_controls(logger) -> None:
    """Logs and prints the keyboard control scheme once at startup."""
    controls = (
        "Controls: [q] Quit  |  [s] Save screenshot  |  "
        "[l] Toggle landmark IDs  |  [b] Toggle bounding box  |  "
        "[g] Toggle gesture debug overlay"
    )
    logger.info(controls)
    print(controls)


def main() -> int:
    """
    Runs the full Phase 1 real-time hand-tracking loop.

    Returns:
        Process exit code: 0 on clean exit, 1 on error (so this can be
        wired into shell scripts / CI later without extra glue code).
    """
    logger = setup_logging(config.paths.log_dir, config.log_level)
    logger.info("=== AR Gesture Controlled System - Phase 1 starting ===")

    fps_meter = FPS()

    try:
        # `with Camera(...)` guarantees cam.release() runs even if an
        # exception is thrown anywhere inside this block.
        with Camera(config.camera, logger) as camera:
            tracker = HandTracker(config.hand_tracker, config.display, logger)
            recognizer = GestureRecognizer(config.gesture, logger)  # Phase 2 addition
            print_controls(logger)

            try:
                while True:
                    ret, frame = camera.read()
                    if not ret:
                        logger.warning("Empty frame received, skipping this iteration.")
                        continue

                    # Core Phase 1 pipeline step: detect hands + get an
                    # annotated frame ready for display.
                    hands_data, annotated_frame = tracker.process(frame)

                    # --- Phase 2 addition: gesture recognition ---
                    # Runs on the primary (first-detected) hand only; feeds
                    # straight off Phase 1's own HandData output.
                    primary_hand = hands_data[0] if hands_data else None
                    gesture_result = recognizer.recognize(primary_hand)
                    if config.gesture.show_gesture_debug:
                        recognizer.draw_overlay(annotated_frame, gesture_result)
                    # --- end Phase 2 addition ---

                    current_fps = fps_meter.update()
                    if config.display.show_fps:
                        draw_text_with_background(
                            annotated_frame,
                            f"FPS: {current_fps:.1f}",
                            (10, 30),
                        )

                    draw_text_with_background(
                        annotated_frame,
                        f"Hands detected: {len(hands_data)}",
                        (10, 60),
                    )

                    cv2.imshow(config.display.window_name, annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit key pressed. Shutting down.")
                        break
                    elif key == ord("s"):
                        save_screenshot(annotated_frame, config.paths.screenshot_dir, logger)
                    elif key == ord("l"):
                        config.display.show_landmark_ids = not config.display.show_landmark_ids
                        logger.info("Landmark ID display toggled: %s", config.display.show_landmark_ids)
                    elif key == ord("b"):
                        config.display.show_bounding_box = not config.display.show_bounding_box
                        logger.info("Bounding box display toggled: %s", config.display.show_bounding_box)
                    elif key == ord("g"):
                        # Phase 2 addition: toggle the gesture debug overlay
                        config.gesture.show_gesture_debug = not config.gesture.show_gesture_debug
                        logger.info("Gesture debug overlay toggled: %s", config.gesture.show_gesture_debug)

                    # Detect the user closing the window via the OS 'X'
                    # button (waitKey alone does not catch this).
                    if cv2.getWindowProperty(config.display.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed by user.")
                        break

            finally:
                # Runs on both normal break AND on exception, ensuring
                # MediaPipe's internal resources are always released.
                tracker.close()
                cv2.destroyAllWindows()

    except CameraError as e:
        logger.error("Camera error: %s", e)
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        # Catch-all so a crash still gets logged with a full traceback
        # (logger.exception includes the stack trace) instead of the
        # user only seeing a raw Python traceback in the console.
        logger.exception("Unexpected fatal error: %s", e)
        print(f"[FATAL ERROR] {e}")
        return 1

    logger.info("=== Application exited cleanly ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
