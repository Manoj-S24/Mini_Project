
import cv2
import time


class Camera:


    def reconnect(self):
        """Try reconnecting up to 5 times."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_running = False

        for attempt in range(1, self.max_retries + 1):

            print(f"Reconnection attempt {attempt}/5")

            self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                success, frame = self.cap.read()

                if success and frame is not None and frame.size > 0:
                    self.is_running = True
                    print("Camera reconnected successfully!")
                    return True

            self.cap.release()
            self.cap = None

            time.sleep(self.retry_delay)

        print("Five attempts failed. Waiting for camera...")
        return False


    def wait_for_camera(self):
        """Periodically check for camera availability."""

        while True:

            print("Searching for camera...")

            if self.reconnect():
                return True

            time.sleep(self.detection_interval)


    def __init__(self, camera_index=0, width=640, height=480):
        self.max_retries = 5
        self.retry_delay = 2
        self.detection_interval = 5
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap = None
        self.is_running = False

   
  
    @staticmethod
    def discover_cameras(max_cameras=5):
        """Find available cameras."""

        available_cameras = []

        for index in range(max_cameras):
            cap = cv2.VideoCapture(index)

            if not cap.isOpened():
                cap.release()
                break

            success, frame = cap.read()

            if success and frame is not None and frame.size > 0:
                available_cameras.append(index)
            else:
                cap.release()
                break

            cap.release()

        return available_cameras


    
    def start(self):
        """Start camera or wait until one becomes available."""
    
        while True:
            self.cap = cv2.VideoCapture(self.camera_index)
    
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
                success, frame = self.cap.read()
    
                if success and frame is not None and frame.size > 0:
                    self.is_running = True
                    print(f"Camera {self.camera_index} started.")
                    return True
    
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
            print("Camera unavailable. Retrying in 5 seconds...")
            time.sleep(self.detection_interval)
    

    
    
    def read_frame(self):
        """Capture frames and recover from camera failures."""

        while self.is_running:

            if self.cap is None:
                if not self.reconnect():
                    self.wait_for_camera()
                continue

            success, frame = self.cap.read()

            if success and frame is not None and frame.size > 0:
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.flip(frame, 1)
                return True, frame

            print("Camera disconnected!")

            if not self.reconnect():
                self.wait_for_camera()

        return False, None


    def get_screen_resolution(self):
        """Detect primary display resolution."""

        import pyautogui

        screen = pyautogui.size()
        return screen.width, screen.height

    def get_frame_dimensions(self, frame):
        if frame is None or frame.size == 0:
            return None

        height, width, channels = frame.shape
        return width, height, channels

    def release(self):
        """Release camera resources."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_running = False
        cv2.destroyAllWindows()

        print("Camera released.")

