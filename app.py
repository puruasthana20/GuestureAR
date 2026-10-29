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
from modules.ar_overlay import AROverlay  # Phase 3 addition
from modules.interaction import InteractionEngine  # Phase 3 addition
from modules.ui_elements import Button, ToggleSwitch, VolumeBar, BrightnessBar, StatusIndicator  # Phase 3 addition


def print_controls(logger) -> None:
    """Logs and prints the keyboard control scheme once at startup."""
    controls = (
        "Controls: [q] Quit  |  [s] Save screenshot  |  "
        "[l] Toggle landmark IDs  |  [b] Toggle bounding box  |  "
        "[g] Toggle gesture debug overlay  |  "
        "[a] Toggle AR overlay  |  [t] Toggle light/dark theme"
    )
    logger.info(controls)
    print(controls)


def build_demo_ar_interface(ar_overlay: AROverlay, logger) -> None:
    """
    Phase 3 addition. Populates `ar_overlay` with a small demo interface:
    two buttons, a toggle, volume/brightness bars and a status badge. A real
    application would build its own element set the same way: create
    elements, optionally attach `on_click`, call `ar_overlay.add_element()`.
    """

    def on_start(_elem):
        logger.info("Demo 'Start' button selected.")
        ar_overlay.add_notification("Start selected", kind="success")

    def on_stop(_elem):
        logger.info("Demo 'Stop' button selected.")
        ar_overlay.add_notification("Stop selected", kind="warning")

    btn_start = Button("btn_start", anchor_x=0.03, anchor_y=0.70, width=0.14, height=0.09, label="Start")
    btn_start.on_click = on_start
    ar_overlay.add_element(btn_start)

    btn_stop = Button("btn_stop", anchor_x=0.19, anchor_y=0.70, width=0.14, height=0.09, label="Stop")
    btn_stop.on_click = on_stop
    ar_overlay.add_element(btn_stop)

    ar_overlay.add_element(ToggleSwitch(
        "toggle_demo", anchor_x=0.03, anchor_y=0.84, width=0.09, height=0.04,
        value=True, label="Demo Toggle"))
    ar_overlay.add_element(VolumeBar("volume", anchor_x=0.90, anchor_y=0.10, width=0.035, height=0.30, value=0.6))
    ar_overlay.add_element(BrightnessBar("brightness", anchor_x=0.95, anchor_y=0.10, width=0.035, height=0.30, value=0.5))
    ar_overlay.add_element(StatusIndicator(
        "status_ar", anchor_x=0.03, anchor_y=0.03, width=0.20, height=0.06,
        label="AR UI", value="Ready", status="good"))


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
            ar_overlay = AROverlay(config.ar, logger)  # Phase 3 addition
            interaction_engine = InteractionEngine(config.ar, logger)  # Phase 3 addition
            build_demo_ar_interface(ar_overlay, logger)  # Phase 3 addition
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

                    # --- Phase 3 addition: AR overlay + interaction ---
                    frame_h, frame_w = annotated_frame.shape[:2]
                    interaction_state = interaction_engine.update(
                        ar_overlay, primary_hand, gesture_result, frame_w, frame_h
                    )
                    annotated_frame = ar_overlay.render(annotated_frame, interaction_state)
                    # --- end Phase 3 addition ---

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
                    elif key == ord("a"):
                        # Phase 3 addition: toggle the AR overlay on/off
                        config.ar.show_ar_overlay = not config.ar.show_ar_overlay
                        logger.info("AR overlay toggled: %s", config.ar.show_ar_overlay)
                    elif key == ord("t"):
                        # Phase 3 addition: switch between dark/light theme
                        ar_overlay.toggle_theme()

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
