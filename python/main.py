
import cv2
import time

from camera import Camera
from hand_tracker import HandTracker

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
    # Camera frames are horizontally flipped
    tracker = HandTracker(max_hands=2, mirrored=True)

    print("Hand tracker initialized successfully!")
    print("Press Q to quit.")

    previous_time = time.perf_counter()

    # Step 4: Main processing loop
    while True:

        success, frame = camera.read_frame()

        if not success:
            continue

        # Step 5: Detect hands
        hands = tracker.process(frame)

        # Step 6: Draw hand landmarks and skeleton
        tracker.draw(frame, hands)

        # Step 7: Calculate FPS
        current_time = time.perf_counter()
        fps = 1 / max(current_time - previous_time, 0.000001)
        previous_time = current_time

        # Step 8: Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Step 9: Display detected hand count
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
