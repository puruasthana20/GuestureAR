# AR-Based Gesture Controlled System for PC and Mobile Interaction
### Phase 1 — Project Foundation & Real-Time Hand Tracking

A production-grade foundation for a hand-gesture-controlled interaction system,
built with OpenCV + MediaPipe. This phase implements a robust, modular,
fully-logged real-time hand-tracking pipeline that later phases (gesture
classification, cursor/media control, AR overlays, PyQt5 GUI, mobile port)
will build directly on top of, without needing to rewrite core plumbing.

---

## 1. Tech Stack

| Component        | Purpose                                             |
|-------------------|------------------------------------------------------|
| Python 3.12       | Core language                                        |
| OpenCV            | Camera I/O, drawing, image processing                |
| MediaPipe         | Real-time hand landmark detection (21 points/hand)   |
| NumPy             | Array/image data handling                            |
| PyQt5             | Reserved for Phase 2+ GUI (not yet wired into app.py) |
| `logging` (stdlib)| Structured console + file logging                    |
| `dataclasses`     | Type-safe config and data structures                 |

---

## 2. Folder Structure

```
GestureAR/
│
├── app.py                 # Entry point — main real-time loop
├── config.py               # Centralized configuration (single source of truth)
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
│
├── assets/                  # Reserved for icons/images used by future GUI
│
├── modules/
│   ├── __init__.py
│   ├── camera.py            # Webcam wrapper (open/read/release, error handling)
│   ├── hand_tracker.py       # MediaPipe Hands wrapper (detection + drawing)
│   ├── utils.py              # Shared helpers (logging setup, bbox math, drawing)
│   └── fps.py                 # Smoothed FPS counter
│
├── logs/                     # Auto-created. One timestamped .log file per run
├── screenshots/               # Auto-created. Screenshots saved with 's' key
│
└── tests/
    ├── __init__.py
    └── test_basic.py          # Unit tests for pure-logic functions (pytest)
```

---

## 3. Installation

### Step 1 — Clone / copy the project
```bash
git clone <your-repo-url> GestureAR
cd GestureAR
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv venv

# Activate:
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on MediaPipe versions:** this project uses MediaPipe's legacy
> `mp.solutions.hands` API. That API ships in MediaPipe `0.10.14` (pinned in
> `requirements.txt`). Newer MediaPipe releases have been observed to change
> or drop top-level `solutions` access on some platforms — if you upgrade
> MediaPipe, verify `mp.solutions.hands` still exists before doing so.

### Step 4 — Run the application
```bash
python app.py
```

### Step 5 — Run tests (optional but recommended)
```bash
pytest tests/ -v
```

---

## 4. Controls

| Key | Action                          |
|-----|----------------------------------|
| `q` | Quit the application              |
| `s` | Save a screenshot to `screenshots/` |
| `l` | Toggle landmark ID numbers on/off |
| `b` | Toggle bounding box on/off        |

---

## 5. File-by-File Explanation

### `config.py`
Defines four `dataclass` groups — `CameraConfig`, `HandTrackerConfig`,
`DisplayConfig`, `PathConfig` — aggregated into one `AppConfig`. A single
instance, `config`, is created at import time and shared by every module.
This means changing `config.camera.frame_width` in one place changes it
everywhere the config is used — there are **no duplicated magic numbers**
anywhere else in the codebase. `PathConfig.__post_init__` auto-creates the
`logs/` and `screenshots/` folders the first time the config is loaded.

### `modules/utils.py`
- **`setup_logging()`** — builds a `logging.Logger` that writes to both the
  console and a timestamped file in `logs/`, so every run leaves an
  auditable trail.
- **`save_screenshot()`** — writes the current annotated frame to
  `screenshots/` with a timestamped filename.
- **`calculate_bounding_box()`** — takes the 21 landmark pixel points and
  returns a padded rectangle `(x_min, y_min, x_max, y_max)` enclosing them.
- **`draw_text_with_background()`** — draws readable white-on-black text,
  used for FPS/hand-count/labels so text stays legible on any background.
- **`clamp()`** — keeps a coordinate inside `[min, max]` so drawing calls
  never receive an off-frame coordinate (which would silently clip or, in
  some OpenCV builds, throw).

### `modules/fps.py`
`FPS` class computes an **exponential moving average** of frame time:
`fps = smoothing * old_fps + (1 - smoothing) * instantaneous_fps`.
This produces a stable, readable on-screen counter instead of a number that
jumps around every frame.

### `modules/camera.py`
`Camera` wraps `cv2.VideoCapture` with:
- `open()` — tries the DirectShow backend first (fast/reliable on Windows),
  falls back to the default backend (Linux/macOS), and raises a clear
  `CameraError` if no backend can open the device.
- `read()` — returns `(success, frame)`, applies the configured horizontal
  flip (mirror effect), and logs a warning on dropped frames instead of
  crashing.
- `release()` — idempotent cleanup.
- Implements the **context-manager protocol** (`__enter__`/`__exit__`) so
  `with Camera(...) as cam:` guarantees the device is released even if an
  exception happens mid-loop.

### `modules/hand_tracker.py`
`HandTracker` wraps `mediapipe.solutions.hands.Hands`:
- Instantiates the MediaPipe `Hands` graph **once** in `__init__` (creating
  it per-frame would reload the model graph every frame — extremely slow).
- `process(frame)` converts BGR→RGB (MediaPipe expects RGB), runs
  inference, and for every detected hand builds a `HandData` object holding
  the handedness label, confidence, 21 pixel-space landmarks, 21
  normalized landmarks, and a bounding box.
- `_draw_hand()` draws the skeleton (via MediaPipe's built-in styled
  drawer), bounding box, handedness label, and optional landmark ID
  numbers.
- `close()` releases MediaPipe's internal resources.

### `app.py`
The orchestration layer:
1. Sets up logging and opens the camera as a context manager.
2. Loops: read frame → `tracker.process()` → overlay FPS/hand-count →
   `cv2.imshow()` → handle key presses.
3. Wraps everything in `try/except/finally` so the camera and MediaPipe
   resources are **always** released, and any crash is logged with a full
   traceback via `logger.exception()`.

### `tests/test_basic.py`
Pure-logic unit tests (bounding box math, clamping, FPS smoothing, config
defaults) that require no webcam or display — safe to run in CI. Camera/
MediaPipe integration is verified manually per the checklist below, since
it requires real hardware.

---

## 6. How MediaPipe Hands Works Internally

MediaPipe's hand solution is a **two-stage pipeline**:

1. **Palm Detector** — a lightweight, single-shot detector (similar in
   spirit to a small object detector like BlazeFace/BlazePalm) scans the
   full frame and outputs oriented bounding boxes around any palms it
   finds. Detecting the *palm* first (a mostly-rigid, easy-to-recognize
   shape) is far more reliable than trying to detect individual thin
   fingers directly in a full-resolution image.
2. **Hand Landmark Model** — for each palm box, the frame is cropped and
   normalized, then fed to a regression CNN that outputs 21 precise 3D
   landmark coordinates (x, y, z) directly, without needing further
   per-frame anchor boxes or NMS.

In **video mode** (`static_image_mode=False`, our default), MediaPipe skips
re-running the palm detector on every single frame. Instead, once a hand is
found, it derives the next frame's crop region directly from the previous
frame's landmarks and tracks it with the (much cheaper) landmark model
alone. The palm detector only re-runs when tracking confidence drops below
`min_tracking_confidence` (e.g. the hand leaves the frame or moves too
fast) — this is the single biggest reason video-mode tracking is so much
faster than re-detecting from scratch every frame.

**Handedness** ("Left"/"Right") is produced by a separate small classifier
head trained alongside the landmark model, and it accounts for the mirrored
nature of a *selfie-view* camera (which is why we flip the frame in
`camera.py` — MediaPipe's handedness labels assume a mirrored/selfie feed).

---

## 7. How the 21 Landmarks Are Detected & Numbered

MediaPipe defines a fixed 21-point skeletal model of the hand:

```
        8   12  16  20
        |   |   |   |
        7   11  15  19
        |   |   |   |
    4   6   10  14  18
    |   |   |   |   |
    3   5   9   13  17
     \   \  |  /   /
      \  \  |  /  /
         \ | / /
           0
```

| ID | Name              | ID | Name               | ID | Name               |
|----|-------------------|----|---------------------|----|---------------------|
| 0  | WRIST              | 7  | INDEX_FINGER_DIP     | 14 | RING_FINGER_PIP      |
| 1  | THUMB_CMC          | 8  | INDEX_FINGER_TIP     | 15 | RING_FINGER_DIP      |
| 2  | THUMB_MCP          | 9  | MIDDLE_FINGER_MCP    | 16 | RING_FINGER_TIP      |
| 3  | THUMB_IP           | 10 | MIDDLE_FINGER_PIP    | 17 | PINKY_MCP            |
| 4  | THUMB_TIP          | 11 | MIDDLE_FINGER_DIP    | 18 | PINKY_PIP            |
| 5  | INDEX_FINGER_MCP   | 12 | MIDDLE_FINGER_TIP    | 19 | PINKY_DIP            |
| 6  | INDEX_FINGER_PIP   | 13 | RING_FINGER_MCP      | 20 | PINKY_TIP            |

This exact ID list is stored in `HandTracker.LANDMARK_NAMES` for use in
logging/debugging and future gesture rules (e.g. Phase 2 can check
`landmarks[8].y < landmarks[6].y` to test "is the index finger raised?").

The model outputs all 21 points **simultaneously** in a single regression
pass (not detected one joint at a time) — it has learned the full hand's
geometric structure from its training data, so occluded or ambiguous
joints are still estimated plausibly from context.

---

## 8. Coordinate Systems Explained

MediaPipe and this project use **three** coordinate representations:

1. **Normalized image coordinates** `(x, y)` — floats in `[0.0, 1.0]`,
   relative to frame width/height. `x=0` is the left edge, `x=1` is the
   right edge (same for y/top-bottom). This is what MediaPipe returns
   natively — it is resolution-independent, which is why it's ideal for
   internal logic in `HandData.landmarks_norm`.
2. **Pixel coordinates** `(x, y)` — integers, computed as
   `px = normalized_x * frame_width`, `py = normalized_y * frame_height`.
   This is what OpenCV's drawing functions (`cv2.circle`, `cv2.rectangle`,
   etc.) require, and what `HandData.landmarks_px` stores.
3. **Normalized depth `z`** — MediaPipe also returns a `z` value per
   landmark, roughly representing depth relative to the wrist, with
   **smaller (more negative) values meaning closer to the camera.** It is
   not metric depth (not in real centimeters) — it's a relative, learned
   estimate — but it's directly useful in Phase 2 for distinguishing
   "pinch toward camera" style gestures. We store it in
   `landmarks_norm[i][2]` for every point, even though Phase 1 doesn't
   yet use it.

The conversion pipeline in `hand_tracker.py`:
```
MediaPipe (normalized 0-1)  →  clamp() → pixel-space (0-w, 0-h)  →  OpenCV drawing
```

---

## 9. Optimization Techniques Used

| Technique | Where | Why |
|---|---|---|
| Reused `Hands` instance across all frames | `hand_tracker.py __init__` | Avoids reloading the TFLite model graph every frame |
| `rgb_frame.flags.writeable = False` before inference | `hand_tracker.py process()` | Documented MediaPipe hint to skip an internal defensive copy |
| Video-mode tracking (`static_image_mode=False`) | `config.py` | Skips full palm re-detection every frame; tracks from prior landmarks instead |
| Small `CAP_PROP_BUFFERSIZE` (=1) | `camera.py` | Prevents OpenCV from queueing stale frames, reducing latency |
| EMA-smoothed FPS counter | `fps.py` | Cheap O(1) update per frame, no rolling window/array needed |
| In-place drawing (`frame` modified directly, not rebuilt) | `utils.py`, `hand_tracker.py` | Avoids unnecessary array copies per overlay element |
| `model_complexity=1` (configurable to `0`) | `config.py` | Lets you trade accuracy for speed on low-end hardware by editing one line |

---

## 10. Testing Checklist

Manual checklist (run `python app.py` and verify each item):

- [ ] Application window opens without errors within a few seconds.
- [ ] Console + `logs/session_*.log` show "Camera opened successfully".
- [ ] Live webcam feed is visible and **mirrored** (raise your right hand,
      it should appear on the right side of the screen, like a mirror).
- [ ] Showing **one hand** draws a skeleton, a colored bounding box, and a
      "Left (NN%)" or "Right (NN%)" label.
- [ ] Showing **two hands** simultaneously tracks and labels both
      independently (test with one left + one right hand visible together).
- [ ] Moving a hand quickly across frame keeps tracking without long freezes.
- [ ] Moving a hand fully out of frame and back in re-acquires tracking.
- [ ] FPS counter (top-left) updates and stays reasonably stable (not
      flickering wildly).
- [ ] "Hands detected: N" count matches what's actually visible.
- [ ] Pressing `l` toggles landmark ID numbers on/off.
- [ ] Pressing `b` toggles bounding boxes on/off.
- [ ] Pressing `s` creates a new PNG in `screenshots/` and logs it.
- [ ] Pressing `q` closes the window and prints "Application exited cleanly".
- [ ] Closing the window with the OS 'X' button also exits cleanly (no hang).
- [ ] Unplugging/disabling the camera mid-run logs a warning instead of
      crashing (frames are just skipped).
- [ ] Running with no webcam connected at all prints a clear `[ERROR]`
      message (not a raw traceback) and exits with code 1.
- [ ] `pytest tests/ -v` — all unit tests pass.

### Expected Output (what you should see)
- A window titled **"AR Gesture System - Phase 1 (Hand Tracking)"**.
- Colored dot-and-line skeleton overlaid on each detected hand.
- A green box around a detected **Right** hand, orange around a **Left** hand.
- Text overlays top-left: `FPS: XX.X` and `Hands detected: N`.
- A text label above each hand reading e.g. `Right (98%)`.

*(Since this is a live webcam application, screenshots depend on your local
camera/hand — use the `s` key while running to capture your own sample
output into `screenshots/` for your project report.)*

---

## 11. Possible Errors & Debugging Guide

| Symptom | Likely Cause | Fix |
|---|---|---|
| `CameraError: Cannot open webcam at index 0` | No camera connected, wrong index, or camera in use by another app | Close Zoom/Teams/other apps using the camera; try `device_index=1` in `config.py`; check OS camera permissions |
| Window opens but feed is black | Camera opened but returning empty frames (driver issue) | Check `logs/session_*.log` for repeated "Failed to read frame" warnings; try a different USB port; update webcam drivers |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | An incompatible MediaPipe version was installed (newer builds have been seen to omit the legacy `solutions` API on some platforms) | `pip install mediapipe==0.10.14` exactly as pinned in `requirements.txt` |
| Very low FPS (<10) | `model_complexity=1` is too heavy for your CPU, or `frame_width/height` too large | Set `model_complexity: int = 0` and/or lower `frame_width`/`frame_height` in `config.py` |
| Hand detected but label always wrong (Left shown as Right) | Camera not mirrored / `flip_horizontal=False` | Ensure `CameraConfig.flip_horizontal = True` (MediaPipe's handedness model assumes a selfie/mirrored view) |
| `ModuleNotFoundError: No module named 'cv2'` / `'mediapipe'` | Dependencies not installed, or wrong virtual environment active | Re-run `pip install -r requirements.txt` inside the activated venv |
| Window doesn't close / app hangs on quit | Rare OpenCV/Qt backend interaction on some Linux window managers | Press `q` in the OpenCV window (not the terminal); if it persists, `Ctrl+C` in terminal is a safe fallback since resources are released in `finally` |
| High CPU fan noise / thermal throttling during long runs | Expected — real-time CV inference is CPU/GPU intensive | Lower `target_fps`, lower resolution, or use `model_complexity=0` for lighter continuous use |
| `pytest` fails on `test_app_config_creates_directories` | Running tests from a read-only filesystem location | Run from a writable working copy of the project, not a read-only mount |

---

## 12. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                              app.py                                  │
│                     (Orchestration / Main Loop)                      │
│                                                                        │
│   ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────┐   │
│   │ config.py│──▶│  modules/    │──▶│ modules/  │──▶│ modules/    │   │
│   │ (settings)│   │  camera.py  │   │hand_tracker│   │   fps.py   │   │
│   └──────────┘   └──────────────┘   └───────────┘   └────────────┘   │
│                          │                  │               │        │
│                          ▼                  ▼               ▼        │
│                    frames (BGR)      HandData[] +      smoothed FPS  │
│                                     annotated_frame                  │
│                          │                  │               │        │
│                          └──────────┬───────┴───────────────┘        │
│                                     ▼                                 │
│                          modules/utils.py                            │
│                (logging, drawing, screenshot, bbox math)             │
│                                     │                                 │
│                                     ▼                                 │
│                          cv2.imshow() window                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 13. Data Flow Diagram

```
 Webcam Hardware
        │
        ▼
 Camera.read()  ──────────────▶  BGR frame (numpy array)
        │
        ▼
 HandTracker.process(frame)
   │
   ├─▶ cv2.cvtColor(BGR → RGB)
   │
   ├─▶ mediapipe Hands.process(rgb_frame)
   │        │
   │        ├─▶ Palm Detector (full-frame, only when needed)
   │        └─▶ Landmark Regressor (21 pts × x,y,z + handedness)
   │
   ├─▶ normalize → pixel coords (× frame width/height)
   │
   ├─▶ calculate_bounding_box(landmarks_px)
   │
   ├─▶ build HandData(label, confidence, landmarks_px,
   │                    landmarks_norm, bounding_box)
   │
   └─▶ _draw_hand() → annotated_frame (skeleton + box + label)
        │
        ▼
 app.py: overlay FPS + hand count
        │
        ▼
 cv2.imshow(window)  ◀────── keyboard input (q / s / l / b)
        │
        ▼
 [optional] save_screenshot() → screenshots/*.png
```

---

## 14. Phase 1 Summary

### ✅ What Has Been Completed
- Full modular project skeleton (config, camera, tracker, utils, fps, app,
  tests) with zero code duplication.
- Robust webcam initialization with dual-backend fallback and clear error
  messages.
- Real-time multi-hand detection (Left, Right, and both simultaneously),
  built on MediaPipe's video-mode tracking for efficiency.
- Full 21-landmark extraction per hand, in both normalized and pixel-space
  coordinates, plus a computed bounding box.
- Live on-screen overlay: skeleton, bounding boxes, handedness + confidence
  labels, landmark IDs (toggleable), FPS counter, hand count.
- Structured logging to console + per-session log files.
- Screenshot capture on demand.
- Runtime-toggleable display options (no restart needed).
- Guaranteed resource cleanup via context managers and `try/finally`, even
  on crashes.
- CI-safe unit test suite covering all pure-logic helper functions.
- Full documentation: architecture diagram, data flow diagram, MediaPipe
  internals, coordinate systems, optimization notes, testing checklist,
  and debugging guide (this README).

### ⚠️ Current Limitations
- No gesture *classification* yet — we detect raw landmarks only; we don't
  yet interpret them as "pinch", "swipe", "fist", etc.
- No actual PC/mobile control actions are wired up (no mouse movement,
  no OS-level input injection) — PyQt5 is installed but not yet used for a
  GUI.
- No AR overlay rendering beyond the debug skeleton/box (no 3D objects,
  no anchoring).
- No persistence/calibration profile per user (e.g. hand size calibration).
- No mobile (Android/iOS) build — this phase targets desktop webcams only.
- Landmark `z` (depth) values are extracted but not yet used for anything.
- Single-camera-only; no multi-camera or IP-camera support yet.

### 🚀 How Phase 2 Will Extend This
1. **Gesture Recognition Layer** — a new `modules/gesture_classifier.py`
   will consume `HandData.landmarks_norm` (already produced in Phase 1) to
   classify gestures (pinch, fist, open palm, swipe, point) using
   geometric rules and/or a small trained classifier.
2. **Action Mapping** — map recognized gestures to OS-level actions
   (cursor movement, click, scroll, volume, media control) via libraries
   like `pyautogui`, feeding off the same `HandData` stream.
3. **PyQt5 GUI** — a settings/dashboard window (calibration, sensitivity,
   gesture-to-action mapping, live preview) built on top of the already
   camera-agnostic `Camera`/`HandTracker` modules — no rework needed there.
4. **AR Overlay Rendering** — richer on-screen graphics (virtual buttons,
   sliders, 3D anchored objects) rendered relative to the existing
   `bounding_box` / landmark positions.
5. **Mobile Port** — the modular boundary between `Camera` (input source)
   and `HandTracker` (processing) means a mobile camera feed can be
   substituted for the desktop `cv2.VideoCapture` source with minimal
   changes to the rest of the pipeline.

Phase 1 deliberately keeps `HandTracker` output in a clean, structured
`HandData` format specifically so Phase 2 can build entirely on top of it
without touching `camera.py`, `hand_tracker.py`, or `config.py` internals.
