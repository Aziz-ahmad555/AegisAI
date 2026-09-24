import cv2
import threading
import time
from ultralytics import YOLO


class VisionStream:
    """
    Manages a single shared webcam capture and runs one of several detection
    modes in a background thread. Multiple browser clients can view the same
    MJPEG stream without opening multiple webcam handles.

    Performance notes (learned the hard way):
    - The capture/inference loop is throttled to a fixed max rate rather than
      running as fast as the hardware allows, so it doesn't consume an entire
      CPU core continuously on a CPU-only machine.
    - Heavy inference is skipped entirely whenever no client is actively
      viewing the /video_feed stream - otherwise the loop kept running full
      YOLO inference in the background forever after the first visit to the
      page, even while the user was on a completely different page, starving
      every other request on the same machine of CPU.
    - Frames are downscaled and JPEG-compressed more aggressively before
      encoding, since the original full-resolution frames were unnecessarily
      large for a live preview and made streaming over a bandwidth-limited
      tunnel (e.g. ngrok's free tier) painfully laggy.
    """

    MODES = ["tracking", "fire_smoke", "fall_detection"]
    TARGET_FPS = 8
    STREAM_WIDTH = 480

    def __init__(self):
        self.general_model = YOLO("yolov8n.pt")
        self.fire_model = YOLO("fire_smoke_model.pt")
        self.pose_model = None

        self.cap = None
        self.lock = threading.RLock()
        self.mode = "tracking"
        self.latest_jpeg = None
        self.latest_info = {"mode": "tracking", "detail": ""}
        self.camera_available = True
        self._running = False
        self._viewer_count = 0
        self._placeholder_jpeg = self._build_placeholder()

    def add_viewer(self):
        with self.lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self.lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    def _has_viewers(self):
        with self.lock:
            return self._viewer_count > 0

    def _build_placeholder(self):
        import numpy as np
        frame = np.zeros((270, 480, 3), dtype="uint8")
        frame[:] = (17, 17, 17)
        cv2.putText(frame, "No camera available", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (61, 163, 232), 2)
        cv2.putText(frame, "This feature requires local hardware", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def _idle_jpeg(self, frame):
        # Downscale even the "just show the camera, no inference" view so it
        # stays cheap to stream while no one is actively watching.
        small = self._resize(frame)
        ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buffer.tobytes() if ok else None

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w <= self.STREAM_WIDTH:
            return frame
        scale = self.STREAM_WIDTH / w
        return cv2.resize(frame, (self.STREAM_WIDTH, int(h * scale)))

    def set_mode(self, mode):
        if mode not in self.MODES:
            return False
        with self.lock:
            self.mode = mode
            if mode == "fall_detection" and self.pose_model is None:
                self.pose_model = YOLO("yolov8n-pose.pt")
        return True

    def _process_tracking(self, frame):
        results = self.general_model.track(frame, imgsz=320, persist=True, verbose=False)
        annotated = results[0].plot()
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"Objects tracked: {count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated, f"{count} objects tracked"

    def _process_fire_smoke(self, frame):
        results = self.fire_model(frame, imgsz=320, conf=0.55, verbose=False)
        annotated = results[0].plot()
        fire_detected = results[0].boxes is not None and len(results[0].boxes) > 0
        label = "FIRE/SMOKE DETECTED" if fire_detected else "Zone clear"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated, label

    def _process_fall_detection(self, frame):
        results = self.pose_model(frame, imgsz=320, verbose=False)
        annotated = results[0].plot()
        fall_detected = False

        for r in results:
            if r.keypoints is None or r.keypoints.conf is None:
                continue
            for kpts_xy, kpts_conf in zip(r.keypoints.xy, r.keypoints.conf):
                if len(kpts_xy) < 17:
                    continue
                needed = [5, 6, 11, 12]
                if any(kpts_conf[i] < 0.5 for i in needed):
                    continue
                ls, rs = kpts_xy[5], kpts_xy[6]
                lh, rh = kpts_xy[11], kpts_xy[12]
                shoulder_x = (ls[0] + rs[0]) / 2
                shoulder_y = (ls[1] + rs[1]) / 2
                hip_x = (lh[0] + rh[0]) / 2
                hip_y = (lh[1] + rh[1]) / 2
                dx = abs(hip_x - shoulder_x)
                dy = abs(hip_y - shoulder_y)
                if dx > dy:
                    fall_detected = True

        label = "POSSIBLE FALL DETECTED" if fall_detected else "Normal posture"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated, label

    def _capture_loop(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(0)

        if not self.cap or not self.cap.isOpened():
            with self.lock:
                self.camera_available = False
                self.latest_info = {
                    "mode": self.mode,
                    "detail": "No camera available - this feature requires local hardware and is not available in the hosted demo.",
                }
            return

        with self.lock:
            self.camera_available = True
        self.latest_info = {"mode": self.mode, "detail": "Waiting for first frame..."}
        consecutive_failures = 0
        frame_interval = 1.0 / self.TARGET_FPS

        while self._running:
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures % 20 == 0:
                    with self.lock:
                        self.latest_info = {
                            "mode": self.mode,
                            "detail": f"ERROR: webcam not delivering frames ({consecutive_failures} failed reads) - close any other app using the camera and restart the server",
                        }
                time.sleep(0.1)
                continue
            consecutive_failures = 0

            with self.lock:
                current_mode = self.mode

            if self._has_viewers():
                try:
                    frame = self._resize(frame)
                    if current_mode == "tracking":
                        annotated, detail = self._process_tracking(frame)
                    elif current_mode == "fire_smoke":
                        annotated, detail = self._process_fire_smoke(frame)
                    elif current_mode == "fall_detection":
                        annotated, detail = self._process_fall_detection(frame)
                    else:
                        annotated, detail = frame, ""

                    ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ok:
                        with self.lock:
                            self.latest_jpeg = buffer.tobytes()
                            self.latest_info = {"mode": current_mode, "detail": detail}
                except Exception as e:
                    with self.lock:
                        self.latest_info = {"mode": current_mode, "detail": f"error: {e}"}
            else:
                # No one is watching - keep the camera warm with a cheap,
                # un-processed idle frame instead of running full inference
                # for nobody, so CPU stays free for everything else.
                with self.lock:
                    self.latest_jpeg = self._idle_jpeg(frame)
                    self.latest_info = {"mode": self.mode, "detail": "Idle (no active viewer)"}

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def get_jpeg(self):
        with self.lock:
            if not self.camera_available:
                return self._placeholder_jpeg
            return self.latest_jpeg

    def get_info(self):
        with self.lock:
            return dict(self.latest_info)

    def generate_mjpeg(self):
        self.add_viewer()
        try:
            while True:
                frame = self.get_jpeg()
                if frame is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(1.0 / self.TARGET_FPS)
        finally:
            self.remove_viewer()
