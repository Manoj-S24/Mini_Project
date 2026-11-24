import ctypes
import math
import time
from pathlib import Path

import cv2
import pyautogui

from camera import Camera
from hand_tracker import HandTracker, select_hand
from gesture_recognizer import GestureRecognizer


# --------------------------------------------------
# Gesture configuration
# --------------------------------------------------

HOLD_THRESHOLD = 0.25       # Hold pinch for 250 ms to start dragging
PINCH_THRESHOLD = 0.05      # Start pinch
RELEASE_THRESHOLD = 0.07    # Release pinch (hysteresis)

# Scrolling configuration
SCROLL_SPEED = 200
WHEEL_DELTA = 120
MOUSEEVENTF_HWHEEL = 0x1000

# C EMA smoothing factor. Keep the value already used in your project.
SMOOTHING_ALPHA = 0.8


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


def horizontal_scroll(direction: str) -> None:
    """Send a horizontal mouse-wheel event on Windows."""
    if direction == "RIGHT":
        delta = WHEEL_DELTA * SCROLL_SPEED
    elif direction == "LEFT":
        delta = -WHEEL_DELTA * SCROLL_SPEED
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
# Main application
# --------------------------------------------------

try:
    # Find an available camera.
    while True:
        available = Camera.discover_cameras()

        if available:
            print("Available cameras:", available)
            break

        print("No camera detected. Retrying in 5 seconds...")
        time.sleep(5)

    # Start camera.
    camera = Camera(camera_index=available[0])
    camera.start()

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

        if not success or frame is None:
            continue

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

                direction = result.scroll_direction

                if direction == "UP":
                    pyautogui.scroll(SCROLL_SPEED)

                elif direction == "DOWN":
                    pyautogui.scroll(-SCROLL_SPEED)

                elif direction in ("LEFT", "RIGHT"):
                    horizontal_scroll(direction)

            else:
                # 4. Detect thumb-index pinch.
                thumb_x, thumb_y = hand.landmarks[4][:2]
                index_x, index_y = hand.landmarks[8][:2]

                distance = math.hypot(
                    index_x - thumb_x,
                    index_y - thumb_y,
                )

                # Pinch begins.
                if not pinching and distance < PINCH_THRESHOLD:
                    pinching = True
                    dragging = False
                    pinch_start_time = time.perf_counter()

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

                # 5. Detect hold-to-drag.
                if pinching and not dragging:
                    elapsed = time.perf_counter() - pinch_start_time

                    if elapsed >= HOLD_THRESHOLD:
                        pyautogui.mouseDown(button="left")
                        dragging = True

            # 6. Display current gesture state.
            if dragging:
                gesture_text = "DRAGGING"
            elif pinching:
                gesture_text = "PINCHING"
            else:
                gesture_text = gesture

            cv2.putText(
                frame,
                f"Gesture: {gesture_text}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Show scroll direction while in scroll mode.
            if gesture == "SCROLL_MODE":
                cv2.putText(
                    frame,
                    f"Scroll: {result.scroll_direction}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            # 7. Move cursor only in MOVE mode, or while pinching/dragging.
            # Do not move the cursor in SCROLL_MODE or NONE mode.
            if gesture == "MOVE" or pinching or dragging:
                tip_x, tip_y = hand.index_tip_normalized()

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

                # Keep the cursor away from the extreme screen corners.
                safe_x = max(1, min(cursor_x.value, screen_width - 2))
                safe_y = max(1, min(cursor_y.value, screen_height - 2))

                pyautogui.moveTo(safe_x, safe_y)

                cv2.putText(
                    frame,
                    f"Cursor: ({cursor_x.value}, {cursor_y.value})",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Tip: ({tip_x:.2f}, {tip_y:.2f})",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
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

            recognizer.reset()
            engine.reset_smoothing()

            cv2.putText(
                frame,
                "Gesture: NO HAND",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # 11. Calculate and display FPS.
        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - previous_time, 0.000001)
        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Hands detected: {len(hands)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 12. Display camera feed.
        cv2.imshow("Universal Gesture Pointer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as error:
    print("Error:", error)

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
