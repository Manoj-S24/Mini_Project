import ctypes
import math
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import pyautogui

from camera import Camera
from hand_tracker import HandTracker, select_hand
from gesture_recognizer import GestureRecognizer


# --------------------------------------------------
# Windows setup (DPI awareness + pyautogui speed)
# --------------------------------------------------

# Make screen coordinates real pixels on scaled (125% / 150%) displays.
# Must run before pyautogui.size(). Harmless if already DPI aware.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# pyautogui sleeps 0.1 s after every call by default, which caps the
# cursor at ~10 updates per second. Disable that delay.
pyautogui.PAUSE = 0


# --------------------------------------------------
# Gesture configuration
# --------------------------------------------------

HOLD_THRESHOLD = 0.25       # Hold pinch for 250 ms to start dragging

# Pinch thresholds are now RATIOS of (thumb-index distance / knuckle width),
# so they no longer depend on how far your hand is from the camera.
# 0.33 / 0.47 roughly match the old 0.05 / 0.07 at a typical distance.
PINCH_THRESHOLD = 0.33      # Start pinch
RELEASE_THRESHOLD = 0.47    # Release pinch (hysteresis)

# Scrolling configuration
SCROLL_SPEED = 200          # wheel units per scroll "tick"
WHEEL_DELTA = 120
MOUSEEVENTF_HWHEEL = 0x1000
H_SCROLL_NOTCHES = 2        # horizontal wheel notches per tick (was 200!)
SCROLL_TICKS_PER_SEC = 6.0  # ticks per second at scroll_amount 1.0

# C EMA smoothing factor. Keep the value already used in your project.
SMOOTHING_ALPHA = 0.8

# Fraction of the camera frame (centered) mapped to the whole screen.
# 1.0 = off (use this if cursor_engine.dll already applies a margin).
# Try 0.7-0.8 so you never have to reach the frame edges.
ACTIVE_REGION = 1.0

# On pinch start, jump back this many cursor updates so the click lands
# where you aimed, not where the index tip drifted while pinching.
PINCH_ANCHOR_FRAMES = 5

# After this many failed frame reads in a row, try to reconnect the camera.
MAX_CAMERA_FAILURES = 90


# --------------------------------------------------
# Windows horizontal scrolling
# --------------------------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)

user32.mouse_event.argtypes = [
    ctypes.c_uint,      # dwFlags
    ctypes.c_int,       # dx
    ctypes.c_int,       # dy
    ctypes.c_uint,      # dwData
    ctypes.c_size_t,    # dwExtraInfo
]
user32.mouse_event.restype = None


def horizontal_scroll(direction: str, ticks: int = 1) -> None:
    """Send a horizontal mouse-wheel event on Windows."""
    if direction == "RIGHT":
        delta = WHEEL_DELTA * H_SCROLL_NOTCHES * ticks
    elif direction == "LEFT":
        delta = -WHEEL_DELTA * H_SCROLL_NOTCHES * ticks
    else:
        return

    # mouse_event expects DWORD data; convert signed delta to DWORD form.
    user32.mouse_event(
        MOUSEEVENTF_HWHEEL,
        0,
        0,
        ctypes.c_uint(delta).value,
        0,
    )


# --------------------------------------------------
# Load and configure the C cursor engine
# --------------------------------------------------

dll_path = Path(__file__).resolve().parent / "cursor_engine.dll"
engine = ctypes.CDLL(str(dll_path))

engine.map_to_screen.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
engine.map_to_screen.restype = None

engine.smooth_coordinates.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
engine.smooth_coordinates.restype = None

engine.reset_smoothing.argtypes = []
engine.reset_smoothing.restype = None


# --------------------------------------------------
# Screen and application state
# --------------------------------------------------

screen_width, screen_height = pyautogui.size()
print(f"Screen resolution: {screen_width}x{screen_height}")

camera = None
tracker = None

recognizer = GestureRecognizer()

pinching = False
dragging = False
pinch_start_time = 0.0

previous_time = time.perf_counter()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def open_camera():
    """Block until a camera is found, then start and return it."""
    while True:
        available = Camera.discover_cameras()

        if available:
            print("Available cameras:", available)
            cam = Camera(camera_index=available[0])
            cam.start()
            return cam

        print("No camera detected. Retrying in 5 seconds...")
        time.sleep(5)


def draw_text(frame, text, origin, scale, color):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def pinch_ratio(hand, frame_w, frame_h):
    """Thumb-tip to index-tip distance divided by knuckle width.

    Measured in pixel-proportional space (not skewed normalized space) and
    scaled by hand size, so it is stable at any distance from the camera.
    """
    lm = hand.landmarks

    def pt(i):
        return (lm[i][0] * frame_w, lm[i][1] * frame_h)

    knuckle_width = math.dist(pt(5), pt(17))
    return math.dist(pt(4), pt(8)) / max(knuckle_width, 1e-6)


def apply_active_region(value):
    """Stretch the central ACTIVE_REGION of the frame to the full 0..1 range."""
    if ACTIVE_REGION >= 1.0:
        return value

    low = (1.0 - ACTIVE_REGION) / 2.0
    return min(max((value - low) / ACTIVE_REGION, 0.0), 1.0)


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():
    global camera, tracker
    global pinching, dragging, pinch_start_time, previous_time

    fps = 0.0
    camera_failures = 0
    frame_size = (0, 0)

    scroll_accumulator = 0.0
    last_scroll_direction = "NONE"

    cursor_history = deque(maxlen=PINCH_ANCHOR_FRAMES)
    pinch_anchor = None     # cursor position frozen at pinch start
    drag_offset = None      # keeps the cursor from jumping when a drag starts

    try:
        # Find an available camera and start it.
        camera = open_camera()

        # Initialize hand tracking.
        # mirrored=True preserves the setting from your existing main.py.
        tracker = HandTracker(max_hands=1, mirrored=True)

        print("Hand tracker initialized successfully!")
        print("Gesture recognizer initialized successfully!")
        print("C cursor engine loaded successfully!")
        print("Cursor smoothing enabled!")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read_frame()

            # Keep the window responsive and recover from a dead camera.
            if not success or frame is None:
                camera_failures += 1

                if camera_failures >= MAX_CAMERA_FAILURES:
                    print("Camera not responding. Reconnecting...")

                    if dragging:
                        pyautogui.mouseUp(button="left")

                    pinching = False
                    dragging = False
                    recognizer.reset()
                    engine.reset_smoothing()

                    try:
                        camera.release()
                    except Exception:
                        pass

                    camera = open_camera()
                    camera_failures = 0

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(0.005)
                continue

            camera_failures = 0

            frame_h, frame_w = frame.shape[:2]

            if (frame_w, frame_h) != frame_size:
                frame_size = (frame_w, frame_h)
                recognizer.set_frame_size(frame_w, frame_h)

            # Frame timing (dt) and smoothed FPS.
            current_time = time.perf_counter()
            dt = max(current_time - previous_time, 0.000001)
            previous_time = current_time

            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

            # 1. Detect and draw hand landmarks.
            hands = tracker.process(frame)
            tracker.draw(frame, hands)

            if hands:
                hand = select_hand(hands)

                # 2. Recognize the hand pose and scrolling movement.
                result = recognizer.process(hand)
                gesture = result.gesture

                # 3. Handle the two-finger scroll mode.
                if gesture == "SCROLL_MODE":

                    # Never leave the mouse button held while switching to scroll.
                    if dragging:
                        pyautogui.mouseUp(button="left")

                    dragging = False
                    pinching = False
                    pinch_anchor = None
                    drag_offset = None
                    cursor_history.clear()

                    direction = result.scroll_direction

                    if direction == "NONE":
                        scroll_accumulator = 0.0
                    else:
                        if direction != last_scroll_direction:
                            scroll_accumulator = 0.0

                        # Joystick-style: further from centre = faster,
                        # independent of FPS.
                        scroll_accumulator += (
                            SCROLL_TICKS_PER_SEC * result.scroll_amount * dt
                        )
                        ticks = int(scroll_accumulator)

                        if ticks > 0:
                            scroll_accumulator -= ticks

                            if direction == "UP":
                                pyautogui.scroll(SCROLL_SPEED * ticks)

                            elif direction == "DOWN":
                                pyautogui.scroll(-SCROLL_SPEED * ticks)

                            elif direction in ("LEFT", "RIGHT"):
                                horizontal_scroll(direction, ticks)

                    last_scroll_direction = direction

                else:
                    scroll_accumulator = 0.0
                    last_scroll_direction = "NONE"

                    # 4. Detect thumb-index pinch (distance relative to hand size).
                    distance = pinch_ratio(hand, frame_w, frame_h)

                    # Pinch begins (a closed fist does not count).
                    if (
                        not pinching
                        and distance < PINCH_THRESHOLD
                        and not recognizer.is_fist(hand)
                    ):
                        pinching = True
                        dragging = False
                        drag_offset = None
                        pinch_start_time = time.perf_counter()

                        # Freeze the cursor where you were aiming a few
                        # frames ago, before the fingers started closing.
                        if cursor_history:
                            pinch_anchor = cursor_history[0]
                            pyautogui.moveTo(*pinch_anchor)
                        else:
                            pinch_anchor = tuple(pyautogui.position())

                    # Pinch ends.
                    elif pinching and distance > RELEASE_THRESHOLD:

                        if dragging:
                            # A held pinch becomes a drag; release the mouse button.
                            pyautogui.mouseUp(button="left")
                        else:
                            # A short pinch becomes a single click.
                            pyautogui.click(button="left")

                        dragging = False
                        pinching = False
                        pinch_anchor = None
                        drag_offset = None

                    # 5. Detect hold-to-drag.
                    if pinching and not dragging:
                        elapsed = time.perf_counter() - pinch_start_time

                        if elapsed >= HOLD_THRESHOLD:
                            pyautogui.mouseDown(button="left")
                            dragging = True
                            drag_offset = None

                # 6. Display current gesture state.
                if dragging:
                    gesture_text = "DRAGGING"
                elif pinching:
                    gesture_text = "PINCHING"
                else:
                    gesture_text = gesture

                draw_text(frame, f"Gesture: {gesture_text}", (20, 100), 0.8, (0, 255, 0))

                # Show scroll direction while in scroll mode.
                if gesture == "SCROLL_MODE":
                    draw_text(
                        frame,
                        f"Scroll: {result.scroll_direction}",
                        (20, 135),
                        0.7,
                        (255, 255, 0),
                    )

                # 7. Move cursor only in MOVE mode, or while dragging.
                # While pinching (before the drag starts) the cursor stays
                # frozen so the click lands where you aimed.
                # Do not move the cursor in SCROLL_MODE or NONE mode.
                if gesture == "MOVE" or pinching or dragging:
                    raw_x, raw_y = hand.index_tip_normalized()
                    tip_x = apply_active_region(raw_x)
                    tip_y = apply_active_region(raw_y)

                    if not (pinching and not dragging):
                        smooth_x = ctypes.c_float()
                        smooth_y = ctypes.c_float()

                        # 8. Apply C dead-zone filtering and EMA smoothing.
                        engine.smooth_coordinates(
                            ctypes.c_float(tip_x),
                            ctypes.c_float(tip_y),
                            ctypes.c_float(SMOOTHING_ALPHA),
                            ctypes.byref(smooth_x),
                            ctypes.byref(smooth_y),
                        )

                        # 9. Map smoothed normalized coordinates to screen pixels.
                        cursor_x = ctypes.c_int()
                        cursor_y = ctypes.c_int()

                        engine.map_to_screen(
                            smooth_x.value,
                            smooth_y.value,
                            screen_width,
                            screen_height,
                            ctypes.byref(cursor_x),
                            ctypes.byref(cursor_y),
                        )

                        target_x = cursor_x.value
                        target_y = cursor_y.value

                        if dragging and pinch_anchor is not None:
                            # First drag frame: remember the gap between the
                            # frozen cursor and the finger, so the cursor
                            # continues from where it is instead of jumping.
                            if drag_offset is None:
                                drag_offset = (
                                    pinch_anchor[0] - target_x,
                                    pinch_anchor[1] - target_y,
                                )

                            target_x += drag_offset[0]
                            target_y += drag_offset[1]

                        # Keep the cursor away from the extreme screen corners.
                        safe_x = max(1, min(target_x, screen_width - 2))
                        safe_y = max(1, min(target_y, screen_height - 2))

                        pyautogui.moveTo(safe_x, safe_y)

                        if not dragging:
                            cursor_history.append((safe_x, safe_y))

                        draw_text(
                            frame,
                            f"Cursor: ({safe_x}, {safe_y})",
                            (20, 170),
                            0.6,
                            (255, 0, 0),
                        )

                    draw_text(
                        frame,
                        f"Tip: ({tip_x:.2f}, {tip_y:.2f})",
                        (20, 200),
                        0.6,
                        (255, 0, 0),
                    )

                else:
                    # Reset C smoothing when not controlling the cursor.
                    engine.reset_smoothing()

            else:
                # 10. Hand lost: release any held mouse button and reset states.
                if dragging:
                    pyautogui.mouseUp(button="left")

                pinching = False
                dragging = False
                pinch_anchor = None
                drag_offset = None
                scroll_accumulator = 0.0
                last_scroll_direction = "NONE"
                cursor_history.clear()

                recognizer.reset()
                engine.reset_smoothing()

                draw_text(frame, "Gesture: NO HAND", (20, 100), 0.8, (0, 0, 255))

            # 11. Display FPS (smoothed) and hand count.
            draw_text(frame, f"FPS: {fps:.1f}", (10, 30), 0.7, (0, 255, 0))
            draw_text(frame, f"Hands detected: {len(hands)}", (10, 60), 0.7, (0, 255, 0))

            # 12. Display camera feed.
            cv2.imshow("Universal Gesture Pointer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print("Error:", error)
        traceback.print_exc()

    finally:
        # Always release the mouse button when the application closes.
        try:
            pyautogui.mouseUp(button="left")
        except Exception:
            pass

        if tracker is not None:
            tracker.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()