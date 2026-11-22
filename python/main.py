
import cv2
import time
import ctypes
from pathlib import Path
import pyautogui

from camera import Camera
from hand_tracker import HandTracker


# Load C DLL
dll_path = Path(__file__).resolve().parent / "cursor_engine.dll"
engine = ctypes.CDLL(str(dll_path))

# Define C function argument and return types
engine.map_to_screen.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]
engine.map_to_screen.restype = None


# Get screen resolution
screen_width, screen_height = pyautogui.size()

print(f"Screen resolution: {screen_width}x{screen_height}")

camera = None
tracker = None

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
    print("Press Q to quit.")

    previous_time = time.perf_counter()

    # Step 4: Main processing loop
    while True:

        success, frame = camera.read_frame()

        if not success:
            continue

        # Step 5: Detect hand
        hands = tracker.process(frame)

        # Step 6: Draw hand landmarks
        tracker.draw(frame, hands)

        # Step 7: Map index fingertip coordinates using C
        if hands:
            hand = hands[0]

            # Index fingertip is landmark 8
            tip_x, tip_y = hand.index_tip_normalized()

            cursor_x = ctypes.c_int()
            cursor_y = ctypes.c_int()

            engine.map_to_screen(
                tip_x,
                tip_y,
                screen_width,
                screen_height,
                ctypes.byref(cursor_x),
                ctypes.byref(cursor_y)
            )
            pyautogui.moveTo(
                cursor_x.value,
                cursor_y.value
            )

            cv2.putText(
                frame,
                f"Cursor: ({cursor_x.value}, {cursor_y.value})",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # Display fingertip position
            cv2.putText(
                frame,
                f"Tip: ({tip_x:.2f}, {tip_y:.2f})",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        # Step 8: Calculate FPS
        current_time = time.perf_counter()
        fps = 1 / max(current_time - previous_time, 0.000001)
        previous_time = current_time

        # Step 9: Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Step 10: Display detected hand count
        cv2.putText(
            frame,
            f"Hands detected: {len(hands)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

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

    cv2.destroyAllWindows()
    print("Application closed.")
