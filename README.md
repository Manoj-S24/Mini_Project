# Universal Camera-Only Gesture Pointing System

Camera in, hand skeleton out, index fingertip tracked as the pointing cursor.

```
python/
  camera.py          # Member 1: camera + frame management
  hand_tracker.py    # Member 2: MediaPipe hand tracking (this branch)
  main.py            # integration: camera feed + skeleton
models/
  hand_landmarker.task
tests/
  test_camera.py
  test_hand_tracker.py
```

## Setup

Use Python 3.10+ in a virtual environment (verified on 3.15; pin 3.10–3.12 if a
MediaPipe wheel is missing for your version):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

The model is already in `models/`. To re-download it:

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Hand tracker usage (Member 2)

```python
from hand_tracker import HandTracker, TipSmoother, map_to_screen

with HandTracker() as tracker:                      # mirrored=True for flipped frames
    hands = tracker.process(frame_bgr)              # [] when tracking is lost
    tracker.draw(frame_bgr, hands)                  # skeleton + red index fingertip
    if hands:
        tip_x, tip_y = hands[0].index_tip_normalized()
        screen = map_to_screen(tip_x, tip_y, 1920, 1080, margin=0.15)
        cursor = TipSmoother(alpha=0.5).update_point(*screen)
```

- `process()` returns one `Hand` per detected hand with 21 normalized landmarks,
  handedness and confidence.
- Landmark 8 is the index fingertip; `index_tip_px()` gives frame pixels,
  `index_tip_normalized()` gives 0..1 coordinates.
- `map_to_screen()` maps the central `[margin, 1 - margin]` region to the whole
  screen, so the cursor reaches every edge and corner while the hand stays in
  view. Lower the margin if the hand cannot reach, raise it if the cursor is too
  twitchy.
- `TipSmoother` is an EMA filter for the cursor; it resets on tracking loss.
  `alpha=0.5` measured ~12% less jitter and ~40% smaller jumps; `0.3` is steadier
  but lags more. Tune live with `--alpha`.
- `select_hand(hands)` returns the highest-confidence hand. Use it for a
  single-hand cursor; handedness labels can flicker on hands cut by the frame
  edge, so do not select by label.
- Handedness is corrected for the raw camera by default. Pass `mirrored=True`
  when you feed horizontally flipped frames (do not do both).
- Camera failures are Member 1's responsibility; this module never touches the camera.

## Live check (webcam)

```bash
python python/live_check.py                     # window, press q
python python/live_check.py --frames 150 --no-window   # scripted run
```

Shows detection rate, FPS, handedness, fingertip, mapped screen position and a
small screen preview. On exit it prints detection %, process time, label counts,
normalized reach, and raw vs smoothed screen jitter. Saves annotated frames to
the temp dir.

## Tests

```bash
pytest tests -q
```

Tests that need MediaPipe or the model skip automatically when unavailable.

## Branches

Work on `dev/nishanth-kkj9`. Never push to `main`; it is a shared integration branch.
