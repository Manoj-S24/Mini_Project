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
    # NEW (optional, defaults to 0.0 so old code keeps working):
    # 0.0 when not scrolling, otherwise 1.0 .. max_scroll_amount.
    # Larger value = hand pushed further past the dead zone = scroll faster.
    scroll_amount: float = 0.0


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
        axis_dominance=1.15,
        hysteresis=15,          # degrees: extended > 150, stays extended until < 135
        debounce_frames=3,      # frames a new gesture must persist before switching
        max_scroll_amount=3.0,  # cap for GestureResult.scroll_amount
        fist_index_ratio=1.0,   # tune: index-tip-to-knuckle / knuckle-width below this = fist
    ):
        self.finger_threshold = finger_threshold
        self.scroll_threshold = scroll_threshold
        self.axis_dominance = axis_dominance
        self.hysteresis = hysteresis
        self.debounce_frames = max(1, int(debounce_frames))
        self.max_scroll_amount = max_scroll_amount
        self.fist_index_ratio = fist_index_ratio

        # Frame size scaling (fixes aspect-ratio distortion of angles).
        # Call set_frame_size(width, height) once you know the frame size.
        self._scale_x = 1.0
        self._scale_y = 1.0

        # Per-finger "currently extended" memory for angle hysteresis.
        self._extended = {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        }

        # Gesture debouncing.
        self._candidate = None
        self._count = 0
        self._stable = "NONE"

        self._scroll_anchor = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_frame_size(self, width, height):
        """Tell the recognizer the camera frame size so angles are computed
        in pixel-proportional space instead of skewed normalized space."""
        if width and height:
            self._scale_x = float(width)
            self._scale_y = float(height)

    def _pt(self, lm, index):
        p = lm[index]
        # MediaPipe z uses roughly the same scale as x.
        return (
            p[0] * self._scale_x,
            p[1] * self._scale_y,
            p[2] * self._scale_x,
        )

    def _angle(self, lm, a, b, c):
        return calculate_angle(
            self._pt(lm, a),
            self._pt(lm, b),
            self._pt(lm, c),
        )

    def _is_extended(self, name, angle):
        was_extended = self._extended[name]
        limit = self.finger_threshold - (self.hysteresis if was_extended else 0)
        state = angle > limit
        self._extended[name] = state
        return state

    # ------------------------------------------------------------------
    # Finger / gesture detection
    # ------------------------------------------------------------------

    def detect_fingers(self, hand):

        lm = hand.landmarks

        thumb_angle = self._angle(lm, 2, 3, 4)
        index_angle = self._angle(lm, 5, 6, 8)
        middle_angle = self._angle(lm, 9, 10, 12)
        ring_angle = self._angle(lm, 13, 14, 16)
        pinky_angle = self._angle(lm, 17, 18, 20)

        return FingerStates(
            thumb=self._is_extended("thumb", thumb_angle),
            index=self._is_extended("index", index_angle),
            middle=self._is_extended("middle", middle_angle),
            ring=self._is_extended("ring", ring_angle),
            pinky=self._is_extended("pinky", pinky_angle),
        )

    def classify_gesture(self, hand):
        """Raw (single-frame) classification. process() debounces it."""

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

    def is_fist(self, hand):
        """True if the hand looks like a closed fist (stateless).

        Used by main.py so a fist is not mistaken for a thumb-index pinch.
        Middle, ring and pinky must be clearly curled AND the index tip must
        be close to its knuckle (a pinch keeps the index finger curved but
        reaching out, a fist tucks it into the palm).
        """
        lm = hand.landmarks
        curled_limit = self.finger_threshold - 2 * self.hysteresis

        for a, b, c in ((9, 10, 12), (13, 14, 16), (17, 18, 20)):
            if self._angle(lm, a, b, c) >= curled_limit:
                return False

        p5 = self._pt(lm, 5)
        p8 = self._pt(lm, 8)
        p17 = self._pt(lm, 17)

        knuckle_width = math.dist(p5[:2], p17[:2])
        tip_to_knuckle = math.dist(p8[:2], p5[:2])

        return tip_to_knuckle / max(knuckle_width, 1e-6) < self.fist_index_ratio

    def _debounce(self, raw):
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1

        if self._count >= self.debounce_frames:
            self._stable = raw

        return self._stable

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, hand):

        gesture = self._debounce(self.classify_gesture(hand))

        if gesture != "SCROLL_MODE":
            self._scroll_anchor = None
            return GestureResult(gesture)

        lm = hand.landmarks

        # Track the midpoint between index and middle fingertips.
        center_x = (lm[8][0] + lm[12][0]) / 2
        center_y = (lm[8][1] + lm[12][1]) / 2

        current_center = (center_x, center_y)

        # First frame establishes the neutral (joystick centre) position.
        if self._scroll_anchor is None:
            self._scroll_anchor = current_center
            return GestureResult("SCROLL_MODE")

        # The anchor stays fixed while in SCROLL_MODE: the further the hand
        # is from it, the faster the scroll (joystick style).
        dx = center_x - self._scroll_anchor[0]
        dy = center_y - self._scroll_anchor[1]

        # Scale movement threshold relative to hand size.
        palm_scale = math.dist(lm[0][:2], lm[9][:2])
        threshold = max(
            self.scroll_threshold,
            palm_scale * 0.10
        )

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx < threshold and abs_dy < threshold:
            return GestureResult("SCROLL_MODE", "NONE", dx, dy)

        direction = "NONE"
        magnitude = 0.0

        # Vertical movement
        if abs_dy >= threshold and abs_dy >= abs_dx * self.axis_dominance:
            direction = "UP" if dy < 0 else "DOWN"
            magnitude = abs_dy

        # Horizontal movement
        elif abs_dx >= threshold and abs_dx >= abs_dy * self.axis_dominance:
            direction = "LEFT" if dx < 0 else "RIGHT"
            magnitude = abs_dx

        if direction == "NONE":
            return GestureResult("SCROLL_MODE", "NONE", dx, dy)

        amount = min(
            self.max_scroll_amount,
            1.0 + (magnitude - threshold) / threshold,
        )

        return GestureResult(
            gesture="SCROLL_MODE",
            scroll_direction=direction,
            delta_x=dx,
            delta_y=dy,
            scroll_amount=amount,
        )

    def reset(self):
        self._scroll_anchor = None
        self._candidate = None
        self._count = 0
        self._stable = "NONE"
        for name in self._extended:
            self._extended[name] = False