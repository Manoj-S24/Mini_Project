
import cv2
import time

from camera import Camera

camera = Camera(camera_index=0, width=640, height=480)

try:
    camera.start()

    previous_time = time.perf_counter()

    while True:
        success, frame = camera.read_frame()

        if not success:
            print("Camera frame unavailable. Stopping...")
            break

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        fps = 1 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        dimensions = camera.get_frame_dimensions(frame)

        if dimensions is None:
            print("Invalid frame dimensions!")
            break

        width, height, channels = dimensions

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

        cv2.imshow("Universal Gesture Pointer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    screen_width, screen_height = camera.get_screen_resolution()

    print("\nSystem Display Information")
    print("--------------------------")
    print(f"Screen Width  : {screen_width}")
    print(f"Screen Height : {screen_height}")
    print(f"Resolution    : {screen_width}x{screen_height}")    
except RuntimeError as error:
    print("Camera error:", error)

finally:
    camera.release()
