
import cv2


class Camera:

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False

    def start(self):
        """Open and initialize the camera."""

        if self.is_running:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Unable to open camera {self.camera_index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.is_running = True

        print("Camera started successfully!")
        return True

    def read_frame(self):
        """Capture, validate and prepare one frame."""

        if not self.is_running or self.cap is None:
            return False, None

        success, frame = self.cap.read()

        # Check whether frame capture succeeded
        if not success or frame is None:
            print("Warning: Unable to capture frame.")
            return False, None

        # Validate frame dimensions
        if frame.size == 0:
            print("Warning: Empty frame received.")
            return False, None

        # Resize frame
        frame = cv2.resize(frame, (self.width, self.height))

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        return True, frame

    def get_frame_dimensions(self, frame):
        """Return frame width, height and channels."""

        if frame is None or frame.size == 0:
            return None

        height, width, channels = frame.shape
        return width, height, channels

    def release(self):
        """Safely release camera resources."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_running = False
        cv2.destroyAllWindows()

        print("Camera released.")

    def get_screen_resolution(self):
        """Detect the primary system display resolution."""
        import pyautogui
        screen_width, screen_height = pyautogui.size()
        return screen_width, screen_height

