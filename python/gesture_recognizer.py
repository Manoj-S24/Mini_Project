
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FingerStates:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool


@dataclass(frozen=True)
class GestureResult:
    gesture: str
    scroll_direction: str = "NONE"
    delta_x: float = 0.0
    delta_y: float = 0.0


def calculate_angle(a, b, c):
    """Calculate the angle ABC using 3D landmark coordinates."""

    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    dot = sum(ba[i] * bc[i] for i in range(3))

    mag_ba = math.sqrt(sum(v * v for v in ba))
    mag_bc = math.sqrt(sum(v * v for v in bc))

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


class GestureRecognizer:

    def __init__(
        self,
        finger_threshold=150,
        scroll_threshold=0.015,
        axis_dominance=1.15
    ):
        self.finger_threshold = finger_threshold
        self.scroll_threshold = scroll_threshold
        self.axis_dominance = axis_dominance

        self._scroll_anchor = None

    def detect_fingers(self, hand):

        lm = hand.landmarks

        index_angle = calculate_angle(lm[5], lm[6], lm[8])
        middle_angle = calculate_angle(lm[9], lm[10], lm[12])
        ring_angle = calculate_angle(lm[13], lm[14], lm[16])
        pinky_angle = calculate_angle(lm[17], lm[18], lm[20])

        return FingerStates(
            thumb=False,
            index=index_angle > self.finger_threshold,
            middle=middle_angle > self.finger_threshold,
            ring=ring_angle > self.finger_threshold,
            pinky=pinky_angle > self.finger_threshold
        )

    def classify_gesture(self, hand):

        fingers = self.detect_fingers(hand)

        if (
            fingers.index
            and fingers.middle
            and not fingers.ring
            and not fingers.pinky
        ):
            return "SCROLL_MODE"

        if (
            fingers.index
            and not fingers.middle
            and not fingers.ring
            and not fingers.pinky
        ):
            return "MOVE"

        return "NONE"

    def process(self, hand):

        gesture = self.classify_gesture(hand)

        if gesture != "SCROLL_MODE":
            self._scroll_anchor = None
            return GestureResult(gesture)

        lm = hand.landmarks

        # Track the midpoint between index and middle fingertips.
        center_x = (lm[8][0] + lm[12][0]) / 2
        center_y = (lm[8][1] + lm[12][1]) / 2

        current_center = (center_x, center_y)

        # First frame establishes the starting position.
        if self._scroll_anchor is None:
            self._scroll_anchor = current_center
            return GestureResult("SCROLL_MODE")

        dx = center_x - self._scroll_anchor[0]
        dy = center_y - self._scroll_anchor[1]

        # Scale movement threshold relative to hand size.
        palm_scale = math.dist(lm[0][:2], lm[9][:2])
        threshold = max(
            self.scroll_threshold,
            palm_scale * 0.10
        )

        if abs(dx) < threshold and abs(dy) < threshold:
            return GestureResult("SCROLL_MODE", "NONE", dx, dy)

        direction = "NONE"

        # Vertical movement
        if abs(dy) >= abs(dx) * self.axis_dominance:
            direction = "UP" if dy < 0 else "DOWN"

        # Horizontal movement
        elif abs(dx) >= abs(dy) * self.axis_dominance:
            direction = "LEFT" if dx < 0 else "RIGHT"

        if direction != "NONE":
            self._scroll_anchor = current_center

        return GestureResult(
            gesture="SCROLL_MODE",
            scroll_direction=direction,
            delta_x=dx,
            delta_y=dy
        )

    def reset(self):
        self._scroll_anchor = None
