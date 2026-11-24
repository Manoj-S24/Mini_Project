
import cv2
import time
import ctypes
from pathlib import Path
import pyautogui
import math
from camera import Camera
from hand_tracker import HandTracker,select_hand
from gesture_recognizer import GestureRecognizer


HOLD_THRESHOLD = 0.25  # 250 milliseconds
# Windows horizontal scroll configuration
WHEEL_DELTA = 120
SCROLL_SPEED = 200
MOUSEEVENTF_HWHEEL = 0x1000

user32 = ctypes.WinDLL("user32", use_last_error=True)

user32.mouse_event.argtypes = [
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
    ctypes.c_size_t
]

user32.mouse_event.restype = None


def horizontal_scroll(direction):
    delta = (
    WHEEL_DELTA * SCROLL_SPEED
    if direction == "RIGHT"
    else -WHEEL_DELTA * SCROLL_SPEED
)

    user32.mouse_event(
        MOUSEEVENTF_HWHEEL,
        0,
        0,
        ctypes.c_uint(delta).value,
        0
    )

pinching = False
dragging = False
pinch_start_time = 0.0


# --------------------------------------------------
# 1. Load C DLL
# --------------------------------------------------

dll_path = Path(__file__).resolve().parent / "cursor_engine.dll"
engine = ctypes.CDLL(str(dll_path))


# --------------------------------------------------
# 2. Configure C coordinate mapping function
# --------------------------------------------------

engine.map_to_screen.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]

engine.map_to_screen.restype = None


# --------------------------------------------------
# 3. Configure C smoothing function
# --------------------------------------------------

engine.smooth_coordinates.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]

engine.smooth_coordinates.restype = None


# --------------------------------------------------
# 4. Get actual screen resolution
# --------------------------------------------------

screen_width, screen_height = pyautogui.size()

print(f"Screen resolution: {screen_width}x{screen_height}")


# --------------------------------------------------
# 5. Initialize application variables
# --------------------------------------------------

camera = None
tracker = None
recognizer = GestureRecognizer()

alpha = 0.8


try:

    # Step 1: Search for an available camera

    while True:

        available = Camera.discover_cameras()

        if available:
            print("Available cameras:", available)
            break

        print("No camera detected. Retrying in 5 seconds...")
        time.sleep(5)


    # Step 2: Initialize and start camera

    camera = Camera(camera_index=available[0])
    camera.start()


    # Step 3: Initialize hand tracker

    tracker = HandTracker(max_hands=1, mirrored=True)

    print("Hand tracker initialized successfully!")
    print("C cursor engine loaded successfully!")
    print("Cursor smoothing enabled!")
    print("Press Q to quit.")


    previous_time = time.perf_counter()


    # --------------------------------------------------
    # 6. Main processing loop
    # --------------------------------------------------
    # Pinch gesture configuration
    PINCH_THRESHOLD = 0.05
    RELEASE_THRESHOLD = 0.07

    pinching = False
    while True:

        success, frame = camera.read_frame()

        if not success:
            continue


        # Step 4: Detect hand

        hands = tracker.process(frame)


        # Step 5: Draw hand landmarks

        tracker.draw(frame, hands)


        
        # Step 6: Process detected hand

        if hands:

            hand = select_hand(hands)

            # Step 6.1: Recognize current gesture
            result = recognizer.process(hand)
            gesture = result.gesture

            # Step 6.2: Process scrolling
            if gesture == "SCROLL_MODE":

                # End any active drag when entering scroll mode
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

                # Step 6.3: Detect pinch
                thumb_x, thumb_y = hand.landmarks[4][:2]
                index_x, index_y = hand.landmarks[8][:2]

                distance = math.sqrt(
                    (index_x - thumb_x) ** 2 +
                    (index_y - thumb_y) ** 2
                )

                if not pinching and distance < PINCH_THRESHOLD:
                    pinching = True
                    dragging = False
                    pinch_start_time = time.perf_counter()

                elif pinching and distance > RELEASE_THRESHOLD:

                    if dragging:
                        pyautogui.mouseUp(button="left")
                        dragging = False

                    else:
                        pyautogui.click(button="left")

                    pinching = False

                # Step 6.4: Detect drag
                if pinching and not dragging:

                    elapsed = time.perf_counter() - pinch_start_time

                    if elapsed >= HOLD_THRESHOLD:
                        pyautogui.mouseDown(button="left")
                        dragging = True

            # Step 6.5: Display current status

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
                2
            )

            if gesture == "SCROLL_MODE":

                cv2.putText(
                    frame,
                    f"Scroll: {result.scroll_direction}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            # Step 6.6: Move cursor only in MOVE mode
            # Also allow movement while pinching or dragging.

            if gesture == "MOVE" or pinching or dragging:

                tip_x, tip_y = hand.index_tip_normalized()

                smooth_x = ctypes.c_float()
                smooth_y = ctypes.c_float()

                engine.smooth_coordinates(
                    ctypes.c_float(tip_x),
                    ctypes.c_float(tip_y),
                    ctypes.c_float(alpha),
                    ctypes.byref(smooth_x),
                    ctypes.byref(smooth_y)
                )

                cursor_x = ctypes.c_int()
                cursor_y = ctypes.c_int()

                engine.map_to_screen(
                    smooth_x.value,
                    smooth_y.value,
                    screen_width,
                    screen_height,
                    ctypes.byref(cursor_x),
                    ctypes.byref(cursor_y)
                )

                safe_x = max(
                    1,
                    min(cursor_x.value, screen_width - 2)
                )

                safe_y = max(
                    1,
                    min(cursor_y.value, screen_height - 2)
                )

                pyautogui.moveTo(safe_x, safe_y)

                cv2.putText(
                    frame,
                    f"Cursor: ({cursor_x.value}, {cursor_y.value})",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Tip: ({tip_x:.2f}, {tip_y:.2f})",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            else:
                engine.reset_smoothing()

        else:

            # No hand detected
            if dragging:
                pyautogui.mouseUp(button="left")

            pinching = False
            dragging = False

            recognizer.reset()
            engine.reset_smoothing()


        # Step 10: Calculate FPS

        current_time = time.perf_counter()

        fps = 1 / max(
            current_time - previous_time,
            0.000001
        )

        previous_time = current_time


        # Step 11: Display FPS

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Step 12: Display detected hand count

        cv2.putText(
            frame,
            f"Hands detected: {len(hands)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Step 13: Display camera window

        cv2.imshow("Universal Gesture Pointer", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as error:

    print("Error:", error)


finally:

    if tracker is not None:
        tracker.close()

    if camera is not None:
        camera.release()
    pyautogui.mouseUp(button="left")
    cv2.destroyAllWindows()

    print("Application closed.")
