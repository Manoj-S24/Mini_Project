
import cv2


class Camera:

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("Camera started successfully!")

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()

        if not success:
            return False, None

        # Resize frame
        frame = cv2.resize(frame, (self.width, self.height))

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        return True, frame

    def get_frame_dimensions(self, frame):
        height, width, channels = frame.shape
        return width, height, channels

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        print("Camera released.")
