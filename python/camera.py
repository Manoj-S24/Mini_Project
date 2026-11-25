import os
import threading
import time

import cv2


def _default_backend():
    """Pick the OpenCV capture backend.

    On Windows, DirectShow opens cameras far faster than the default MSMF
    backend (which can take many seconds). If DirectShow cannot open the
    camera, the default backend is tried automatically.
    """
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def _open_working(index, backend, props=()):
    """Open camera `index` and return it only if it delivers a real frame.

    Tries `backend` first, then OpenCV's default backend. `props` is a list
    of (cv2.CAP_PROP_*, value) pairs applied before the test read.
    Returns an opened VideoCapture, or None.
    """
    backends = [backend]
    if backend != cv2.CAP_ANY:
        backends.append(cv2.CAP_ANY)

    for candidate in backends:
        try:
            cap = cv2.VideoCapture(index, candidate)
        except Exception:
            continue

        if not cap.isOpened():
            cap.release()
            continue

        for prop, value in props:
            cap.set(prop, value)

        # The first frames can fail while the camera warms up.
        for _ in range(5):
            success, frame = cap.read()

            if success and frame is not None and frame.size > 0:
                return cap

            time.sleep(0.05)

        cap.release()

    return None


class Camera:

    def reconnect(self):
        """Try reconnecting up to max_retries (5) times."""

        self._close_capture()
        self.connected = False

        for attempt in range(1, self.max_retries + 1):

            if self._stop_event.is_set():
                return False

            print(f"Reconnection attempt {attempt}/{self.max_retries}")

            cap = self._open_capture()

            if cap is not None:
                self.cap = cap
                self.is_running = True
                self.connected = True
                print("Camera reconnected successfully!")
                return True

            # Interruptible sleep: wakes up immediately on release().
            self._stop_event.wait(self.retry_delay)

        print(f"{self.max_retries} attempts failed. Waiting for camera...")
        return False


    def wait_for_camera(self):
        """Periodically check for camera availability."""

        while not self._stop_event.is_set():

            print("Searching for camera...")

            if self.reconnect():
                return True

            self._stop_event.wait(self.detection_interval)

        return False


    def __init__(
        self,
        camera_index=0,
        width=640,
        height=480,
        fps=30,
        backend=None,       # None = auto (DirectShow on Windows). e.g. cv2.CAP_ANY
        fourcc="MJPG",      # lighter USB bandwidth, usually higher FPS. None = off
    ):
        self.max_retries = 5
        self.retry_delay = 2
        self.detection_interval = 5
        self.max_read_failures = 5      # failed reads in a row before reconnecting
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend if backend is not None else _default_backend()
        self.fourcc = fourcc

        # Kept for compatibility. Frames are no longer dropped to hit this
        # rate; the capture thread always keeps the newest frame instead.
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0

        self.cap = None
        self.is_running = False
        self.connected = False          # True while frames are arriving

        # Capture thread state.
        self._thread = None
        self._stop_event = threading.Event()
        self._cond = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._returned_id = 0


    # ------------------------------------------------------------------
    # Opening / closing
    # ------------------------------------------------------------------

    def _capture_props(self):
        props = []

        if self.fourcc:
            props.append(
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            )

        props.extend([
            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
            (cv2.CAP_PROP_FPS, self.fps),
            (cv2.CAP_PROP_BUFFERSIZE, 1),   # ignored by backends without it
        ])

        return props

    def _open_capture(self):
        return _open_working(
            self.camera_index,
            self.backend,
            self._capture_props(),
        )

    def _close_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

            self.cap = None


    @staticmethod
    def discover_cameras(max_cameras=5):
        """Find available cameras."""

        available_cameras = []
        misses = 0

        for index in range(max_cameras):

            cap = _open_working(index, _default_backend())

            if cap is None:
                misses += 1

                # Allow one gap (e.g. a virtual camera at index 0 that does
                # not work) but stop after two misses in a row so startup
                # stays fast.
                if misses >= 2:
                    break

                continue

            misses = 0
            cap.release()
            available_cameras.append(index)

        return available_cameras


    def start(self):
        """Start camera or wait until one becomes available."""

        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop_event.clear()

        while not self._stop_event.is_set():

            cap = self._open_capture()

            if cap is not None:
                self.cap = cap
                self.is_running = True
                self.connected = True

                print(f"Camera {self.camera_index} started.")
                self._report_resolution(cap)

                self._thread = threading.Thread(
                    target=self._capture_loop,
                    name="CameraCapture",
                    daemon=True,
                )
                self._thread.start()
                return True

            print("Camera unavailable. Retrying in 5 seconds...")
            self._stop_event.wait(self.detection_interval)

        return False

    def _report_resolution(self, cap):
        try:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            return

        if actual_w and actual_h and (actual_w, actual_h) != (self.width, self.height):
            print(
                f"Warning: camera delivers {actual_w}x{actual_h}; frames will "
                f"be resized to {self.width}x{self.height} (aspect ratio may change)."
            )


    # ------------------------------------------------------------------
    # Capture thread
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Read frames as fast as the camera delivers them and keep only the
        newest one, so the main loop never works on stale, buffered frames
        and never freezes while the camera is reconnecting."""

        failures = 0

        while not self._stop_event.is_set():

            cap = self.cap

            if cap is None:
                self.connected = False

                if not self.reconnect():
                    self.wait_for_camera()

                failures = 0
                continue

            try:
                success, frame = cap.read()
            except Exception:
                success, frame = False, None

            if not success or frame is None or frame.size == 0:
                failures += 1

                if failures >= self.max_read_failures:
                    print("Camera disconnected!")
                    self.connected = False
                    self._close_capture()
                    failures = 0
                else:
                    time.sleep(0.01)

                continue

            failures = 0
            self.connected = True

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            frame = cv2.flip(frame, 1)

            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()


    def read_frame(self, timeout=0.1):
        """Return (True, newest_new_frame), or (False, None) if no new frame
        arrived within `timeout` seconds (camera busy, reconnecting or
        released). Never blocks for longer than `timeout`, so the caller's
        window stays responsive."""

        if not self.is_running:
            return False, None

        with self._cond:
            if self._frame_id == self._returned_id:
                self._cond.wait(timeout)

            if self._frame_id == self._returned_id or self._frame is None:
                return False, None

            self._returned_id = self._frame_id
            self.last_frame_time = time.perf_counter()

            # The capture thread replaces the array every frame and never
            # modifies it afterwards, so handing it out without a copy is safe.
            return True, self._frame


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_screen_resolution(self):
        """Detect primary display resolution."""

        import pyautogui

        screen = pyautogui.size()
        return screen.width, screen.height

    def get_frame_dimensions(self, frame):
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        return width, height, channels

    def release(self):
        """Release camera resources."""

        self._stop_event.set()

        with self._cond:
            self._cond.notify_all()

        thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

        self._thread = None

        self._close_capture()

        self.is_running = False
        self.connected = False

        with self._cond:
            self._frame = None

        cv2.destroyAllWindows()

        print("Camera released.")