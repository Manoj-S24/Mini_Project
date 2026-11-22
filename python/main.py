
import cv2
import time
import ctypes
import math
import pyautogui

from pathlib import Path
from camera import Camera
from hand_tracker import HandTracker, select_hand


# --------------------------------------------------
# 1. Gesture configuration
# --------------------------------------------------

HOLD_THRESHOLD = 0.25       # 250 milliseconds
PINCH_THRESHOLD = 0.05
RELEASE_THRESHOLD = 0.07

pinching = False
dragging = False
pinch_start_time = 0.0


# --------------------------------------------------
# 2. Load C DLL
# --------------------------------------------------

dll_path = Path(__file__).resolve().parent / "cursor_engine.dll"
engine = ctypes.CDLL(str(dll_path))


# --------------------------------------------------
# 3. Configure C coordinate mapping function
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
# 4. Configure C smoothing function
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
# 5. Configure C smoothing reset function
# --------------------------------------------------

engine.reset_smoothing.argtypes = []
engine.reset_smoothing.restype = None


# --------------------------------------------------
# 6. Get actual screen resolution
# --------------------------------------------------

screen_width, screen_height = pyautogui.size()

print(f"Screen resolution: {screen_width}x{screen_height}")


# --------------------------------------------------
# 7. Initialize application variables
# --------------------------------------------------

camera = None
tracker = None

alpha = 0.5


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
    # 8. Main processing loop
    # --------------------------------------------------

    while True:

        success, frame = camera.read_frame()

        if not success:
            continue


        # Step 1: Detect hand

        hands = tracker.process(frame)


        # Step 2: Draw hand landmarks

        tracker.draw(frame, hands)


        # Step 3: Process detected hand

        if hands:

            hand = select_hand(hands)

            # Get thumb and index fingertip coordinates

            thumb_x, thumb_y = hand.landmarks[4][:2]
            index_x, index_y = hand.landmarks[8][:2]


            # Calculate distance between fingertips

            distance = math.sqrt(
                (index_x - thumb_x) ** 2 +
                (index_y - thumb_y) ** 2
            )


            # Step 4: Detect pinch and release

            if not pinching and distance < PINCH_THRESHOLD:

                pinching = True
                dragging = False
                pinch_start_time = time.perf_counter()


            elif pinching and distance > RELEASE_THRESHOLD:

                # Pinch released

                if dragging:

                    # Stop dragging

                    pyautogui.mouseUp(button="left")
                    dragging = False

                else:

                    # Short pinch: single click

                    pyautogui.click(button="left")

                pinching = False


            # Step 5: Detect hold-to-drag

            if pinching and not dragging:

                elapsed = time.perf_counter() - pinch_start_time

                if elapsed >= HOLD_THRESHOLD:

                    pyautogui.mouseDown(button="left")
                    dragging = True


            # Step 6: Display gesture state

            if dragging:
                gesture_text = "DRAGGING"

            elif pinching:
                gesture_text = "PINCHING"

            else:
                gesture_text = "RELEASE"


            cv2.putText(
                frame,
                f"Gesture: {gesture_text}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if pinching else (255, 255, 255),
                2
            )


            # Step 7: Get index fingertip coordinates

            tip_x, tip_y = hand.index_tip_normalized()

            print(f"X: {tip_x:.4f} | Y: {tip_y:.4f}")


            # Step 8: Prepare smoothed coordinates

            smooth_x = ctypes.c_float()
            smooth_y = ctypes.c_float()


            # Step 9: Apply C smoothing

            engine.smooth_coordinates(
                ctypes.c_float(tip_x),
                ctypes.c_float(tip_y),
                ctypes.c_float(alpha),
                ctypes.byref(smooth_x),
                ctypes.byref(smooth_y)
            )


            # Step 10: Map coordinates using C

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


            # Step 11: Move Windows cursor

            # Keep cursor away from the four screen corners

            safe_x = max(1, min(cursor_x.value, screen_width - 2))
            safe_y = max(1, min(cursor_y.value, screen_height - 2))

            pyautogui.moveTo(safe_x, safe_y)


            # Step 12: Display mapped cursor coordinates

            cv2.putText(
                frame,
                f"Cursor: ({cursor_x.value}, {cursor_y.value})",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


            # Step 13: Display fingertip coordinates

            cv2.putText(
                frame,
                f"Tip: ({tip_x:.2f}, {tip_y:.2f})",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


        else:

            # Stop dragging if hand tracking is lost

            if dragging:
                pyautogui.mouseUp(button="left")

            # Reset gesture states

            pinching = False
            dragging = False

            # Reset C cursor smoothing

            engine.reset_smoothing()


        # Step 14: Calculate FPS

        current_time = time.perf_counter()

        fps = 1 / max(
            current_time - previous_time,
            0.000001
        )

        previous_time = current_time


        # Step 15: Display FPS

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Step 16: Display detected hand count

        cv2.putText(
            frame,
            f"Hands detected: {len(hands)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Step 17: Display camera window

        cv2.imshow("Universal Gesture Pointer", frame)


        # Quit application when Q is pressed

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except Exception as error:

    print("Error:", error)


finally:

    # Release mouse button when application closes

    pyautogui.mouseUp(button="left")


    # Close hand tracker

    if tracker is not None:
        tracker.close()


    # Release camera

    if camera is not None:
        camera.release()


    # Close OpenCV windows

    cv2.destroyAllWindows()

    print("Application closed.")
