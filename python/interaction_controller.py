from __future__ import annotations

import ctypes
import math
import time
from pathlib import Path

from gesture_contract import GestureState


# ============================================================
# EXISTING C CURSOR ENGINE ADAPTER
# ============================================================

class CursorEngineAdapter:
    """
    Python adapter for the existing cursor_engine.dll.

    Does not compile, replace, or modify cursor_engine.c.
    """

    def __init__(self, dll_path=None):

        if dll_path is None:
            dll_path = (
                Path(__file__).resolve().parent
                / "cursor_engine.dll"
            )

        dll_path = Path(dll_path)

        if not dll_path.is_file():
            raise FileNotFoundError(
                f"Cursor engine DLL not found: {dll_path}"
            )

        self.dll = ctypes.CDLL(str(dll_path))

        try:
            self.dll.smooth_coordinates.argtypes = [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.dll.smooth_coordinates.restype = None

            self.dll.map_to_screen.argtypes = [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            self.dll.map_to_screen.restype = None

            self.dll.reset_smoothing.argtypes = []
            self.dll.reset_smoothing.restype = None

        except AttributeError as error:
            raise RuntimeError(
                "The DLL does not expose the expected cursor functions"
            ) from error

    def map_and_smooth(
        self,
        x,
        y,
        alpha,
        screen_width,
        screen_height,
    ):

        if not all(map(math.isfinite, (x, y, alpha))):
            raise ValueError("Cursor inputs must be finite")

        if not 0.0 < alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1")

        if screen_width <= 2 or screen_height <= 2:
            raise ValueError("Invalid screen dimensions")

        # Keep coordinates within the normalized range.
        x = max(0.0, min(float(x), 1.0))
        y = max(0.0, min(float(y), 1.0))

        smooth_x = ctypes.c_float()
        smooth_y = ctypes.c_float()

        # Existing C smoothing function.
        self.dll.smooth_coordinates(
            ctypes.c_float(x),
            ctypes.c_float(y),
            ctypes.c_float(alpha),
            ctypes.byref(smooth_x),
            ctypes.byref(smooth_y),
        )

        screen_x = ctypes.c_int()
        screen_y = ctypes.c_int()

        # Existing C mapping function.
        self.dll.map_to_screen(
            smooth_x.value,
            smooth_y.value,
            screen_width,
            screen_height,
            ctypes.byref(screen_x),
            ctypes.byref(screen_y),
        )

        # Preserve the original application's safety margin.
        safe_x = max(1, min(screen_x.value, screen_width - 2))
        safe_y = max(1, min(screen_y.value, screen_height - 2))

        return safe_x, safe_y

    def reset_smoothing(self):
        self.dll.reset_smoothing()


# ============================================================
# MOUSE BACKEND
# ============================================================

class MouseBackend:
    """Isolates PyAutoGUI operations from interaction logic."""

    def __init__(self):

        import pyautogui

        self.pg = pyautogui

        # Keep the emergency corner fail-safe enabled.
        self.pg.FAILSAFE = True

        # Remove PyAutoGUI's default delay between operations.
        self.pg.PAUSE = 0

    def size(self):
        screen = self.pg.size()
        return screen.width, screen.height

    def move(self, x, y):
        self.pg.moveTo(x, y)

    def down(self, button):
        self.pg.mouseDown(button=button)

    def up(self, button):
        self.pg.mouseUp(button=button)

    def click(self, button):
        self.pg.click(button=button)

    def scroll(self, amount):
        self.pg.scroll(amount)


# ============================================================
# INTERACTION CONTROLLER
# ============================================================

class InteractionController:

    def __init__(
        self,
        cursor_engine,
        mouse=None,
        *,
        hold_threshold=0.250,
        gesture_confidence=0.60,
        right_click_cooldown=0.35,
        scroll_gain=100,
        max_scroll_steps=4,
    ):

        if not math.isfinite(hold_threshold) or hold_threshold <= 0:
            raise ValueError("hold_threshold must be positive")

        if (
            not math.isfinite(gesture_confidence)
            or not 0.0 <= gesture_confidence <= 1.0
        ):
            raise ValueError("gesture_confidence must be in [0, 1]")

        if (
            not math.isfinite(right_click_cooldown)
            or right_click_cooldown < 0
        ):
            raise ValueError("right_click_cooldown cannot be negative")

        if not math.isfinite(scroll_gain) or scroll_gain <= 0:
            raise ValueError("scroll_gain must be positive")

        if (
            not isinstance(max_scroll_steps, int)
            or isinstance(max_scroll_steps, bool)
            or max_scroll_steps < 1
        ):
            raise ValueError("max_scroll_steps must be a positive integer")

        self.cursor_engine = cursor_engine
        self.mouse = mouse if mouse is not None else MouseBackend()

        self.screen_width, self.screen_height = self.mouse.size()

        if self.screen_width <= 2 or self.screen_height <= 2:
            raise ValueError("Invalid screen dimensions")

        self.hold_threshold = hold_threshold
        self.gesture_confidence = gesture_confidence
        self.right_click_cooldown = right_click_cooldown
        self.scroll_gain = scroll_gain
        self.max_scroll_steps = max_scroll_steps

        # Cursor state.
        self.last_tip = None
        self.last_timestamp = None
        self.last_cursor = None

        # Left-click and drag state.
        self.left_active = False
        self.pinch_started_at = None
        self.dragging = False

        # Right-click state.
        self.right_armed = True
        self.last_right_click = float("-inf")

        # Scroll state.
        self.scroll_remainder = 0.0

        # Prevent an old gesture from firing after tracking recovery.
        self.await_neutral = False

        self.closed = False

    # --------------------------------------------------------
    # ADAPTIVE CURSOR SMOOTHING
    # --------------------------------------------------------

    def calculate_alpha(self, tip, timestamp):

        if self.last_tip is None:
            return 0.75

        dt = timestamp - self.last_timestamp

        if dt <= 0:
            raise ValueError("Timestamps must increase")

        dx = tip[0] - self.last_tip[0]
        dy = tip[1] - self.last_tip[1]

        speed = math.hypot(dx, dy) / dt

        # Slow movement: more smoothing.
        # Fast movement: less lag.
        ratio = min(speed / 1.0, 1.0)

        return 0.25 + (0.75 - 0.25) * ratio

    def move_cursor(self, fingertip, timestamp):

        alpha = self.calculate_alpha(
            fingertip,
            timestamp,
        )

        position = self.cursor_engine.map_and_smooth(
            fingertip[0],
            fingertip[1],
            alpha,
            self.screen_width,
            self.screen_height,
        )

        # Avoid unnecessary Windows mouse operations.
        if position != self.last_cursor:
            self.mouse.move(*position)
            self.last_cursor = position

        self.last_tip = fingertip
        self.last_timestamp = timestamp

        return position

    # --------------------------------------------------------
    # DRAG RELEASE SAFETY
    # --------------------------------------------------------

    def release_drag(self):

        if self.dragging:
            self.mouse.up("left")

        self.dragging = False

    # --------------------------------------------------------
    # CANCEL ACTIVE INTERACTIONS
    # --------------------------------------------------------

    def cancel_interaction(self):

        self.release_drag()

        self.left_active = False
        self.pinch_started_at = None

        self.right_armed = False
        self.scroll_remainder = 0.0

        self.await_neutral = True

    # --------------------------------------------------------
    # LEFT PINCH / CLICK / DRAG
    # --------------------------------------------------------

    def process_left(self, active, timestamp):

        if active:

            if not self.left_active:
                self.left_active = True
                self.pinch_started_at = timestamp

            elapsed = timestamp - self.pinch_started_at

            if (
                not self.dragging
                and elapsed >= self.hold_threshold
            ):
                self.mouse.down("left")
                self.dragging = True

            return

        if self.left_active:

            if self.dragging:
                self.release_drag()

            else:
                # Short pinch produces exactly one click.
                self.mouse.click("left")

            self.left_active = False
            self.pinch_started_at = None

    # --------------------------------------------------------
    # RIGHT CLICK
    # --------------------------------------------------------

    def process_right(self, active, timestamp):

        if not active:
            self.right_armed = True
            return

        if not self.right_armed:
            return

        # Disarm on activation even if cooldown suppresses the click.
        # A new activation requires a neutral frame.
        self.right_armed = False

        if (
            timestamp - self.last_right_click
            < self.right_click_cooldown
        ):
            return

        self.mouse.click("right")
        self.last_right_click = timestamp

    # --------------------------------------------------------
    # SCROLL
    # --------------------------------------------------------

    def process_scroll(self, active, delta_y):

        if not active:
            self.scroll_remainder = 0.0
            return

        self.scroll_remainder += delta_y * self.scroll_gain

        raw_steps = math.trunc(self.scroll_remainder)

        if raw_steps == 0:
            return

        steps = max(
            -self.max_scroll_steps,
            min(raw_steps, self.max_scroll_steps),
        )

        # Discard excess whole steps to prevent delayed bursts.
        self.scroll_remainder -= raw_steps

        # PyAutoGUI: positive scroll = up.
        # Positive hand movement is down, so reverse the sign.
        self.mouse.scroll(-steps)

    # --------------------------------------------------------
    # MAIN FRAME UPDATE
    # --------------------------------------------------------

    def update(
        self,
        *,
        hand_present,
        fingertip,
        gestures: GestureState,
        timestamp=None,
    ):

        if self.closed:
            raise RuntimeError("Controller is closed")

        if not isinstance(gestures, GestureState):
            raise TypeError("gestures must be a GestureState")

        now = (
            time.perf_counter()
            if timestamp is None
            else float(timestamp)
        )

        if not math.isfinite(now):
            raise ValueError("Invalid timestamp")

        # Tracking lost.
        if not hand_present or fingertip is None:

            self.cancel_interaction()

            self.last_tip = None
            self.last_timestamp = None
            self.last_cursor = None

            self.scroll_remainder = 0.0

            self.cursor_engine.reset_smoothing()

            return None

        # Validate fingertip coordinates.
        if (
            len(fingertip) != 2
            or not all(math.isfinite(float(v)) for v in fingertip)
        ):
            raise ValueError("Invalid fingertip coordinates")

        if (
            self.last_timestamp is not None
            and now <= self.last_timestamp
        ):
            raise ValueError("Timestamps must be strictly increasing")

        # Cursor movement remains independent of gesture actions.
        cursor = self.move_cursor(
            (float(fingertip[0]), float(fingertip[1])),
            now,
        )

        # Ignore uncertain gesture decisions.
        if gestures.confidence < self.gesture_confidence:
            self.cancel_interaction()
            return cursor

        # Require a neutral frame after tracking loss or cancellation.
        if self.await_neutral:

            neutral = not (
                gestures.left_pinch_active
                or gestures.right_click_active
                or gestures.scroll_active
            )

            if neutral:
                self.await_neutral = False
                self.right_armed = True

            return cursor

        # Priority 1: left pinch and drag.
        if gestures.left_pinch_active:

            self.process_left(True, now)

            self.right_armed = False
            self.scroll_remainder = 0.0

            return cursor

        # Suppress other actions on the left-pinch release frame.
        if self.left_active:

            self.process_left(False, now)

            self.right_armed = not gestures.right_click_active
            self.scroll_remainder = 0.0

            return cursor

        # Priority 2: right click.
        if gestures.right_click_active:

            self.process_right(True, now)
            self.scroll_remainder = 0.0

            return cursor

        self.process_right(False, now)

        # Priority 3: scrolling.
        self.process_scroll(
            gestures.scroll_active,
            gestures.scroll_delta_y,
        )

        return cursor

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    def close(self):

        if self.closed:
            return

        try:
            self.release_drag()
        finally:
            self.left_active = False
            self.pinch_started_at = None
            self.scroll_remainder = 0.0

            self.cursor_engine.reset_smoothing()
            self.closed = True