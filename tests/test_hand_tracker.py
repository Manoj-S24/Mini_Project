"""Tests for hand_tracker. MediaPipe/cv2/numpy are required; skipped otherwise."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from hand_tracker import (  # noqa: E402
    HAND_CONNECTIONS,
    INDEX_FINGER_TIP,
    LABEL_FONT_SCALE,
    LABEL_FONT_THICKNESS,
    Hand,
    HandTracker,
    TipSmoother,
    _FrameClock,
    label_origin,
    map_to_screen,
    select_hand,
    swap_handedness,
)


def make_hand(handedness="Right", score=0.9):
    # Landmarks laid out on a grid so every point is distinct.
    landmarks = tuple(((i % 5) / 4, (i // 5) / 4, 0.0) for i in range(21))
    return Hand(landmarks=landmarks, handedness=handedness, score=score)


def test_connections_reference_valid_landmarks():
    assert len(HAND_CONNECTIONS) == 21
    for start, end in HAND_CONNECTIONS:
        assert 0 <= start < 21 and 0 <= end < 21
        assert start != end


def test_px_scales_and_clamps():
    hand = make_hand()
    assert hand.px(0, 640, 480) == (0, 0)
    assert hand.px(4, 640, 480) == (639, 0)
    assert hand.px(20, 640, 480) == (0, 479)

    out_of_bounds = Hand(landmarks=((-0.5, 1.5, 0.0),) * 21, handedness="Left", score=1.0)
    assert out_of_bounds.px(0, 640, 480) == (0, 479)


def test_index_tip_is_landmark_8():
    hand = make_hand()
    assert INDEX_FINGER_TIP == 8
    assert hand.index_tip_px(640, 480) == hand.px(8, 640, 480)


def test_draw_empty_hands_leaves_frame_unchanged():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    HandTracker.draw(frame, [])
    assert np.array_equal(frame, original)


def test_draw_skeleton_and_highlight_tip():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand = make_hand()
    HandTracker.draw(frame, [hand])

    assert frame.any(), "drawing must modify the frame"
    tip_x, tip_y = hand.index_tip_px(640, 480)
    assert tuple(frame[tip_y, tip_x]) == (0, 0, 255), "index tip must be highlighted"


def test_px_rounds_to_nearest_pixel():
    hand = Hand(landmarks=((0.005, 0.005, 0.0),) * 21, handedness="Right", score=1.0)
    assert hand.px(0, 100, 100) == (1, 1)


def test_label_origin_stays_inside_frame():
    (text_width, text_height), _ = cv2.getTextSize(
        "Right 1.00", cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
    )
    for wrist in [(0, 0), (639, 0), (0, 479), (639, 479)]:
        x, y = label_origin("Right 1.00", wrist, 640, 480)
        assert 0 <= x <= 640 - text_width - 4
        assert text_height + 4 <= y <= 476


def test_select_hand_prefers_highest_score():
    weak = make_hand(score=0.4)
    strong = make_hand(score=0.95)
    assert select_hand([weak, strong]) is strong
    assert select_hand([]) is None


def test_swap_handedness_labels():
    assert swap_handedness("Left") == "Right"
    assert swap_handedness("Right") == "Left"
    assert swap_handedness("Unknown") == "Unknown"


def test_map_to_screen_margin_covers_whole_screen():
    assert map_to_screen(0.15, 0.15, 1920, 1080) == (0, 0)
    assert map_to_screen(0.85, 0.85, 1920, 1080) == (1919, 1079)
    assert map_to_screen(0.0, 1.0, 1920, 1080) == (0, 1079)
    assert map_to_screen(0.5, 0.5, 1920, 1080) == (960, 540)
    assert map_to_screen(0.1, 0.1, 1920, 1080, margin=0.1) == (0, 0)
    with pytest.raises(ValueError, match="margin"):
        map_to_screen(0.5, 0.5, 100, 100, margin=0.6)


def test_tip_smoother_resets_on_hand_loss():
    hand = make_hand()
    smoother = TipSmoother(alpha=1.0)
    first = smoother.update(hand, 640, 480)
    assert first == hand.index_tip_px(640, 480)
    assert smoother.update(None, 640, 480) is None
    assert smoother.update(hand, 640, 480) == first
    assert smoother.update_point(10, 10) == (10, 10)
    assert smoother.update_point(20, 20) == (20, 20)
    with pytest.raises(ValueError, match="alpha"):
        TipSmoother(alpha=0.0)


def test_handedness_swapped_for_raw_camera_frames():
    mp = pytest.importorskip("mediapipe")
    from types import SimpleNamespace

    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            category = SimpleNamespace(category_name="Left", score=0.9)
            return SimpleNamespace(hand_landmarks=[landmarks], handedness=[[category]])

    def tracker(mirrored):
        instance = object.__new__(HandTracker)
        instance._mp = mp
        instance._landmarker = FakeLandmarker()
        instance._clock = _FrameClock()
        instance._mirrored = mirrored
        return instance

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert tracker(mirrored=False).process(frame)[0].handedness == "Right"
    assert tracker(mirrored=True).process(frame)[0].handedness == "Left"


def test_frame_clock_is_strictly_increasing():
    clock = _FrameClock(now_ms=lambda: 100)
    stamps = [clock.next() for _ in range(3)]
    assert stamps == [100, 101, 102]


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="hand_landmarker.task"):
        HandTracker(model_path=tmp_path / "missing.task")


def test_blank_frame_reports_no_hands():
    pytest.importorskip("mediapipe")
    from hand_tracker import MODEL_PATH

    if not MODEL_PATH.is_file():
        pytest.skip("hand_landmarker.task not downloaded")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with HandTracker() as tracker:
        hands = tracker.process(frame)
        tracker.draw(frame, hands)
    assert hands == []


def test_grayscale_frame_is_accepted():
    pytest.importorskip("mediapipe")
    from hand_tracker import MODEL_PATH

    if not MODEL_PATH.is_file():
        pytest.skip("hand_landmarker.task not downloaded")

    gray = np.zeros((480, 640), dtype=np.uint8)
    with HandTracker() as tracker:
        assert tracker.process(gray) == []


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        HandTracker.process(None, np.zeros((0, 0, 3), dtype=np.uint8))
