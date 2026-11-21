
import cv2
import time

from camera import Camera

camera = Camera()

try:
    camera.start()

    previous_time = time.perf_counter()

    while True:
        success, frame = camera.read_frame()

        if not success:
            print("Failed to capture frame!")
            break

        # Calculate FPS
        current_time = time.perf_counter()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Get frame dimensions
        width, height, channels = camera.get_frame_dimensions(frame)

        # Display information
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Resolution: {width}x{height}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("Universal Gesture Pointer - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
