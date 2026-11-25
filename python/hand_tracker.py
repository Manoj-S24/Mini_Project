"""Hand tracking and 21-landmark detection (Member 2).

Public API:
    tracker = HandTracker()                  # loads models/hand_landmarker.task
    hands = tracker.process(frame_bgr)       # [] when tracking is lost
    tracker.draw(frame_bgr, hands)           # draws skeleton in place
    tracker.close()

Landmarks are normalized (x, y, z) in [0, 1]. Use Hand.px() / index_tip_px()
to convert them to frame pixels, and map_to_screen() to turn the fingertip into
full-screen cursor coordinates.

Handedness: MediaPipe assumes selfie-mirrored input. Pass mirrored=True when
you feed horizontally flipped frames; the raw cv2 camera frame needs the
default mirrored=False, which swaps MediaPipe's labels to the true hand.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

INDEX_FINGER_TIP = 8
LABEL_FONT_SCALE = 0.6
LABEL_FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe hand topology: thumb, index, middle, ring, pinky, palm base.
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
        x, y, _ = self.landmarks[index]
        return (
            min(max(int(x * width + 0.5), 0), width - 1),
            min(max(int(y * height + 0.5), 0), height - 1),
        )

    def index_tip_px(self, width: int, height: int) -> tuple[int, int]:
        """Index fingertip (landmark 8) in pixels. The pointing cursor."""
        return self.px(INDEX_FINGER_TIP, width, height)

    def index_tip_normalized(self) -> tuple[float, float]:
        """Index fingertip (landmark 8) as normalized (x, y) in [0, 1]."""
        x, y, _ = self.landmarks[INDEX_FINGER_TIP]
        return (x, y)


def select_hand(hands: list[Hand]) -> Hand | None:
    """Most confident hand, for a single-hand cursor. None when empty."""
    return max(hands, key=lambda hand: hand.score) if hands else None


def swap_handedness(label: str) -> str:
    """Left <-> Right. MediaPipe labels assume a mirrored selfie image."""
    return {"Left": "Right", "Right": "Left"}.get(label, label)


def map_to_screen(
    x: float,
    y: float,
    screen_width: int,
    screen_height: int,
    margin: float = 0.15,
) -> tuple[int, int]:
    """Normalized fingertip -> screen pixel covering the whole screen.

    The central [margin, 1 - margin] region maps to the full screen, so the
    cursor reaches every edge and corner while the hand stays in view.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")

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
        """Smooth an arbitrary cursor point, e.g. mapped screen coordinates."""
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
    """Place the label next to the wrist but fully inside the frame."""
    (text_width, text_height), _ = cv2.getTextSize(
        text, FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
    )
    x = min(max(wrist_px[0] + 8, 0), max(width - text_width - 4, 0))
    y = min(max(wrist_px[1] - 8, text_height + 4), max(height - 4, text_height + 4))
    return (x, y)


class _FrameClock:
    """Strictly increasing millisecond timestamps for MediaPipe VIDEO mode."""

    def __init__(self, now_ms=None):
        self._now_ms = now_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._last = -1

    def next(self) -> int:
        timestamp = max(self._now_ms(), self._last + 1)
        self._last = timestamp
        return timestamp


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _alternate_model_paths() -> list[Path]:
    """Other places the model usually lives, for projects that keep
    hand_tracker.py in the same folder as main.py (flat layout) or that are
    started from a different working directory."""
    return [
        Path(__file__).resolve().parent / "models" / MODEL_PATH.name,
        Path.cwd() / "models" / MODEL_PATH.name,
    ]


def _resolve_model_path(model_path: str | Path) -> Path:
    """Return an existing model file, or raise FileNotFoundError.

    An explicitly passed path is used exactly as given. Only the default
    MODEL_PATH falls back to the alternate locations above.
    """
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path

    also_looked: list[Path] = []
    if model_path == MODEL_PATH:
        for candidate in _alternate_model_paths():
            if candidate.is_file():
                return candidate
            if candidate != model_path and candidate not in also_looked:
                also_looked.append(candidate)

    message = (
        f"Hand landmarker model not found: {model_path}\n"
        "Download it with:\n"
        "  curl -L -o models/hand_landmarker.task " + _MODEL_URL
    )
    if also_looked:
        message += "\nAlso looked in:\n" + "\n".join(f"  {p}" for p in also_looked)
    raise FileNotFoundError(message)


class HandTracker:
    """MediaPipe Hand Landmarker wrapper with drawing helpers."""

    _closed = False

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mirrored: bool = False,
    ):
        model_path = _resolve_model_path(model_path)

        # Imported here so the module (and its tests) load without MediaPipe.
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise ImportError(
                "MediaPipe (Tasks API) is required. Install or upgrade it with:\n"
                "  pip install -U mediapipe"
            ) from exc

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self._clock = _FrameClock()
        self._mirrored = mirrored

    def process(self, frame_bgr) -> list[Hand]:
        """Detect hands in a BGR frame. Returns [] when tracking is lost."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("process() needs a non-empty BGR frame")

        if self._closed:
            raise RuntimeError("HandTracker is closed")

        channels = 1 if frame_bgr.ndim == 2 else frame_bgr.shape[2]
        if channels == 1:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        elif channels == 4:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._clock.next())

        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            category = handedness[0]
            label = category.category_name or "Unknown"
            if not self._mirrored:
                label = swap_handedness(label)
            hands.append(
                Hand(
                    landmarks=tuple((p.x, p.y, p.z) for p in landmarks),
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
        """Draw skeleton and highlighted index fingertip in place. Returns frame."""
        height, width = frame_bgr.shape[:2]

        for hand in hands:
            points = [hand.px(i, width, height) for i in range(len(hand.landmarks))]

            for start, end in HAND_CONNECTIONS:
                cv2.line(frame_bgr, points[start], points[end], connection_color, 2, cv2.LINE_AA)

            for index, point in enumerate(points):
                if index != INDEX_FINGER_TIP:
                    cv2.circle(frame_bgr, point, 4, point_color, -1, cv2.LINE_AA)

            tip = points[INDEX_FINGER_TIP]
            cv2.circle(frame_bgr, tip, 10, tip_color, 2, cv2.LINE_AA)
            cv2.circle(frame_bgr, tip, 4, tip_color, -1, cv2.LINE_AA)

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
        """Release the MediaPipe model. Safe to call more than once."""
        if self._closed:
            return

        self._closed = True
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()