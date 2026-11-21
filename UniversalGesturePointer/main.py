
import cv2
from camera import Camera

camera = Camera()

try:
    camera.start()

    while True:
        success, frame = camera.read_frame()

        if not success:
            print("Failed to capture frame!")
            break

        cv2.imshow("Universal Gesture Pointer - Camera", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
