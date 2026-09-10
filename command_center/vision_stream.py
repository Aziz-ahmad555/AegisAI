import cv2
import threading
import time
from ultralytics import YOLO


class VisionStream:
    """
    Manages a single shared webcam capture and runs one of several detection
    modes (reusing Phase 1-3 logic) in a background thread. Multiple browser
    clients can view the same MJPEG stream without opening multiple webcam
    handles, which would fail on most hardware (only one process can hold
    a webcam device at a time).
    """

    MODES = ["tracking", "fire_smoke", "fall_detection"]

    def __init__(self):
        self.general_model = YOLO("yolov8n.pt")
        self.fire_model = YOLO("fire_smoke_model.pt")
        self.pose_model = None  # loaded lazily only if fall_detection mode is used

        self.cap = None
        self.lock = threading.RLock()
        self.mode = "tracking"
        self.latest_jpeg = None
        self.latest_info = {"mode": "tracking", "detail": ""}
        self._running = False

    def set_mode(self, mode):
        if mode not in self.MODES:
            return False
        with self.lock:
            self.mode = mode
            if mode == "fall_detection" and self.pose_model is None:
                self.pose_model = YOLO("yolov8n-pose.pt")
        return True

    def _process_tracking(self, frame):
        results = self.general_model.track(frame, imgsz=416, persist=True, verbose=False)
        annotated = results[0].plot()
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"Objects tracked: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated, f"{count} objects tracked"

    def _process_fire_smoke(self, frame):
        results = self.fire_model(frame, imgsz=416, conf=0.55, verbose=False)
        annotated = results[0].plot()
        fire_detected = results[0].boxes is not None and len(results[0].boxes) > 0
        label = "FIRE/SMOKE DETECTED" if fire_detected else "Zone clear"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated, label

    def _process_fall_detection(self, frame):
        results = self.pose_model(frame, imgsz=416, verbose=False)
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
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated, label

    def _capture_loop(self):
        # CAP_DSHOW is the more reliable backend for webcams on Windows;
        # the default backend can silently fail to deliver frames even
        # when isOpened() reports True.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.latest_info = {"mode": self.mode, "detail": "ERROR: could not open webcam - check no other app/script is using it"}
            return

        self.latest_info = {"mode": self.mode, "detail": "Waiting for first frame..."}
        consecutive_failures = 0

        while self._running:
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

            try:
                if current_mode == "tracking":
                    annotated, detail = self._process_tracking(frame)
                elif current_mode == "fire_smoke":
                    annotated, detail = self._process_fire_smoke(frame)
                elif current_mode == "fall_detection":
                    annotated, detail = self._process_fall_detection(frame)
                else:
                    annotated, detail = frame, ""
            except Exception as e:
                annotated, detail = frame, f"error: {e}"

            ok, buffer = cv2.imencode(".jpg", annotated)
            if ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_info = {"mode": current_mode, "detail": detail}

        self.cap.release()

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_info(self):
        with self.lock:
            return dict(self.latest_info)

    def generate_mjpeg(self):
        while True:
            frame = self.get_jpeg()
            if frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.05)
