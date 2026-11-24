import time
import cv2

from camera import Camera
from hand_tracker import HandTracker, select_hand

from gesture_contract import GestureState
from interaction_controller import (
    CursorEngineAdapter,
    InteractionController,
    MouseBackend,
)


def run(recognizer, camera_index=0):
    """
    Run the gesture-controlled cursor application.

    recognizer must implement:
        update(hand, timestamp) -> GestureState
    """

    if recognizer is None or not callable(
        getattr(recognizer, "update", None)
    ):
        raise TypeError(
            "A gesture recognizer with an update() method is required"
        )

    camera = None
    tracker = None
    controller = None

    try:
        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        camera = Camera(camera_index=camera_index)
        camera.start()

        tracker = HandTracker(
            max_hands=1,
            mirrored=True,
        )

        cursor_engine = CursorEngineAdapter()
        mouse = MouseBackend()

        controller = InteractionController(
            cursor_engine,
            mouse,
        )

        print("Camera initialized.")
        print("Hand tracker initialized.")
        print("Existing C cursor engine loaded.")
        print("Interaction controller initialized.")
        print("Press Q to quit.")

        previous_time = time.perf_counter()

        # ----------------------------------------------------
        # MAIN PROCESSING LOOP
        # ----------------------------------------------------

        while True:

            success, frame = camera.read_frame()

            if not success:
                continue

            # Hand detection.
            hands = tracker.process(frame)

            # Draw landmarks.
            tracker.draw(frame, hands)

            # Select the most confident detected hand.
            hand = select_hand(hands)

            timestamp = time.perf_counter()

            # IMPORTANT:
            # Update recognizer even when hand=None.
            gesture_state = recognizer.update(
                hand,
                timestamp,
            )

            if not isinstance(gesture_state, GestureState):
                raise TypeError(
                    "Recognizer must return GestureState"
                )

            fingertip = (
                hand.index_tip_normalized()
                if hand is not None
                else None
            )

            # Cursor and interaction processing.
            cursor_position = controller.update(
                hand_present=hand is not None,
                fingertip=fingertip,
                gestures=gesture_state,
                timestamp=timestamp,
            )

            # ------------------------------------------------
            # DEBUG INFORMATION
            # ------------------------------------------------

            fps = 1.0 / max(
                timestamp - previous_time,
                1e-6,
            )

            previous_time = timestamp

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
                f"Hands: {len(hands)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture_state.label}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if cursor_position is not None:

                cv2.putText(
                    frame,
                    f"Cursor: {cursor_position}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            if gesture_state.debug_message:

                cv2.putText(
                    frame,
                    gesture_state.debug_message[:70],
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

            cv2.imshow(
                "Universal Gesture Pointer",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # ----------------------------------------------------
        # SAFE RESOURCE CLEANUP
        # ----------------------------------------------------

        if controller is not None:
            controller.close()

        if tracker is not None:
            tracker.close()

        if camera is not None:
            camera.release()

        close_recognizer = getattr(
            recognizer,
            "close",
            None,
        )

        if callable(close_recognizer):
            close_recognizer()

        cv2.destroyAllWindows()