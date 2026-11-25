"""Hand tracking and 21-landmark detection (Member 2).

Public API:
    tracker = HandTracker()
    hands = tracker.process(frame_bgr)
    tracker.draw(frame_bgr, hands)
    tracker.close()

Landmarks: normalized (x, y, z).
Use Hand.px() / index_tip_px() for frame pixel coordinates.
Use map_to_screen() for full-screen cursor coordinates.

Handedness:
MediaPipe assumes selfie-mirrored input.
Pass mirrored=True for horizontally flipped frames.
Raw OpenCV camera frames use mirrored=False by default.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from pathlib import Path

import cv2

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

INDEX_FINGER_TIP = 8
LABEL_FONT_SCALE = 0.6
LABEL_FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class Hand:
    """One detected hand: 21 normalized landmarks plus handedness."""

    landmarks: tuple[tuple[float, float, float], ...]
    handedness: str
    score: float

    def px(self, index: int, width: int, height: int) -> tuple[int, int]:
        """Landmark index -> pixel coordinates clamped to the frame."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        x, y, _ = self.landmarks[index]

        return (
            min(max(int(x * width + 0.5), 0), width - 1),
            min(max(int(y * height + 0.5), 0), height - 1),
        )

    def index_tip_px(self, width: int, height: int) -> tuple[int, int]:
        """Index fingertip (landmark 8) in pixels."""
        return self.px(INDEX_FINGER_TIP, width, height)

    def index_tip_normalized(self) -> tuple[float, float]:
        """Index fingertip (landmark 8) normalized coordinates."""
        x, y, _ = self.landmarks[INDEX_FINGER_TIP]
        return (x, y)


def select_hand(hands: list[Hand]) -> Hand | None:
    """Most confident hand, for a single-hand cursor."""
    return max(hands, key=lambda hand: hand.score) if hands else None


def swap_handedness(label: str) -> str:
    """Left <-> Right."""
    return {"Left": "Right", "Right": "Left"}.get(label, label)


def map_to_screen(
    x: float,
    y: float,
    screen_width: int,
    screen_height: int,
    margin: float = 0.15,
) -> tuple[int, int]:
    """Normalized fingertip -> full-screen coordinates."""

    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("Screen dimensions must be positive")

    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Coordinates must be finite numbers")

    span = 1.0 - 2.0 * margin

    nx = min(max((x - margin) / span, 0.0), 1.0)
    ny = min(max((y - margin) / span, 0.0), 1.0)

    return (
        int(nx * (screen_width - 1) + 0.5),
        int(ny * (screen_height - 1) + 0.5),
    )


class TipSmoother:
    """EMA cursor smoothing; resets when the hand is lost."""

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

        self._alpha = alpha
        self._last: tuple[int, int] | None = None

    def update(self, hand: Hand | None, width: int, height: int):
        """Smoothed fingertip pixels, or None when there is no hand."""
        if hand is None:
            return self.reset()

        return self.update_point(*hand.index_tip_px(width, height))

    def update_point(self, x: int, y: int) -> tuple[int, int]:
        """Smooth an arbitrary cursor point."""
        if self._last is None:
            self._last = (x, y)
        else:
            a = self._alpha

            self._last = (
                int(a * x + (1 - a) * self._last[0] + 0.5),
                int(a * y + (1 - a) * self._last[1] + 0.5),
            )

        return self._last

    def reset(self):
        self._last = None
        return None


def label_origin(text, wrist_px, width, height):
    """Place label next to wrist while staying inside the frame."""

    (text_width, text_height), _ = cv2.getTextSize(
        text, FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
    )

    x = min(max(wrist_px[0] + 8, 0), max(width - text_width - 4, 0))

    y = min(
        max(wrist_px[1] - 8, text_height + 4),
        max(height - 4, text_height + 4),
    )

    return (x, y)


class _FrameClock:
    """Strictly increasing millisecond timestamps for VIDEO mode."""

    def __init__(self, now_ms=None):
        self._now_ms = now_ms or (
            lambda: time.monotonic_ns() // 1_000_000
        )
        self._last = -1

    def next(self) -> int:
        timestamp = max(self._now_ms(), self._last + 1)
        self._last = timestamp
        return timestamp


class HandTracker:
    """MediaPipe Hand Landmarker wrapper with drawing helpers."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mirrored: bool = False,
    ):

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {model_path}\n"
                "Download it with:\n"
                "  curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/"
                "hand_landmarker.task"
            )

        if max_hands < 1:
            raise ValueError("max_hands must be at least 1")

        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_presence_confidence", min_presence_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        # Import here so tests can load the module without MediaPipe.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp

        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

        self._clock = _FrameClock()
        self._mirrored = mirrored
        self._closed = False

    def process(self, frame_bgr) -> list[Hand]:
        """Detect hands in a BGR frame. Returns [] when tracking is lost."""

        if self._closed:
            raise RuntimeError("HandTracker is closed")

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("process() needs a non-empty frame")

        if frame_bgr.dtype != "uint8":
            raise ValueError("Frame must have uint8 dtype")

        if frame_bgr.ndim == 2:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)

        elif frame_bgr.ndim == 3:
            channels = frame_bgr.shape[2]

            if channels == 3:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            elif channels == 4:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)

            else:
                raise ValueError(
                    f"Unsupported frame channel count: {channels}"
                )

        else:
            raise ValueError("Frame must be grayscale, BGR or BGRA")

        # Ensure contiguous RGB memory for MediaPipe.
        rgb = rgb.copy()

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self._landmarker.detect_for_video(
            image,
            self._clock.next(),
        )

        hands = []

        for landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness,
        ):
            category = handedness[0]

            label = category.category_name or "Unknown"

            if not self._mirrored:
                label = swap_handedness(label)

            hands.append(
                Hand(
                    landmarks=tuple(
                        (p.x, p.y, p.z) for p in landmarks
                    ),
                    handedness=label,
                    score=float(category.score),
                )
            )

        return hands

    @staticmethod
    def draw(
        frame_bgr,
        hands: list[Hand],
        *,
        point_color=(0, 255, 0),
        connection_color=(255, 255, 255),
        tip_color=(0, 0, 255),
    ):
        """Draw skeleton and highlighted fingertip in place."""

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("draw() needs a non-empty frame")

        height, width = frame_bgr.shape[:2]

        for hand in hands:
            points = [
                hand.px(i, width, height)
                for i in range(len(hand.landmarks))
            ]

            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame_bgr,
                    points[start],
                    points[end],
                    connection_color,
                    2,
                    cv2.LINE_AA,
                )

            for index, point in enumerate(points):
                if index != INDEX_FINGER_TIP:
                    cv2.circle(
                        frame_bgr,
                        point,
                        4,
                        point_color,
                        -1,
                        cv2.LINE_AA,
                    )

            tip = points[INDEX_FINGER_TIP]

            cv2.circle(
                frame_bgr,
                tip,
                10,
                tip_color,
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame_bgr,
                tip,
                4,
                tip_color,
                -1,
                cv2.LINE_AA,
            )

            label = f"{hand.handedness} {hand.score:.2f}"

            cv2.putText(
                frame_bgr,
                label,
                label_origin(label, points[0], width, height),
                FONT,
                LABEL_FONT_SCALE,
                connection_color,
                LABEL_FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame_bgr

    def close(self) -> None:
        """Release the MediaPipe tracker."""
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()