import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Leave CPU headroom for the web server and the browser: by default PyTorch
# grabs every core for a single inference, which makes the rest of the app
# stutter while a frame is being processed.
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


class VisionStream:
    """
    Low-latency webcam pipeline shared by all browser viewers.

    Three stages, each on its own OS thread, connected by "latest value wins"
    slots rather than queues - so a slow stage drops stale frames instead of
    building up a backlog:

      grabber   -> reads the camera as fast as it delivers, keeps only the
                   newest frame. Draining the driver buffer continuously is
                   what prevents the multi-second lag: previously the loop
                   read one frame, spent ~200 ms on inference, then read the
                   *next buffered* (already old) frame, so delay accumulated.
      inference -> always processes the newest frame, runs the active model,
                   JPEG-encodes the annotated result.
      mjpeg     -> each viewer's generator wakes on a condition as soon as a
                   new JPEG is ready, instead of polling on a fixed sleep.

    Inference is skipped entirely while nobody is viewing /video_feed.
    """

    MODES = ["tracking", "fire_smoke", "fall_detection"]
    STREAM_WIDTH = 480
    INFER_SIZE = 320
    JPEG_QUALITY = 70
    IDLE_FPS = 4

    def __init__(self):
        self.general_model = YOLO("yolov8n.pt")
        self.fire_model = YOLO("fire_smoke_model.pt")
        self.pose_model = None

        self.lock = threading.RLock()
        self.mode = "tracking"
        self.camera_available = True
        self._running = False
        self._viewer_count = 0

        # Newest raw frame from the grabber.
        self._frame = None
        self._frame_id = 0
        self._frame_time = 0.0
        self._frame_ready = threading.Condition(self.lock)

        # Newest encoded output for viewers.
        self._jpeg = None
        self._jpeg_id = 0
        self._jpeg_ready = threading.Condition(self.lock)

        self._info = {"mode": self.mode, "detail": "Starting camera...", "fps": 0.0, "latency_ms": 0}
        self._placeholder_jpeg = self._build_placeholder()

    # ----- viewers ---------------------------------------------------------

    def add_viewer(self):
        with self.lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self.lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    def _has_viewers(self):
        with self.lock:
            return self._viewer_count > 0

    # ----- helpers ---------------------------------------------------------

    def _build_placeholder(self):
        frame = np.zeros((270, 480, 3), dtype="uint8")
        frame[:] = (17, 17, 17)
        cv2.putText(frame, "No camera available", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (61, 163, 232), 2)
        cv2.putText(frame, "This feature requires local hardware", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w <= self.STREAM_WIDTH:
            return frame
        scale = self.STREAM_WIDTH / w
        return cv2.resize(frame, (self.STREAM_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

    def _encode(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        return buffer.tobytes() if ok else None

    def _warm_up(self):
        # The first YOLO call on CPU is several times slower than steady state
        # (lazy graph setup, memory allocation). Pay that cost at startup
        # rather than on the first frame a viewer sees.
        dummy = np.zeros((self.INFER_SIZE, self.INFER_SIZE, 3), dtype="uint8")
        self.general_model.predict(dummy, imgsz=self.INFER_SIZE, verbose=False)
        self.fire_model.predict(dummy, imgsz=self.INFER_SIZE, verbose=False)

    # ----- mode ------------------------------------------------------------

    def set_mode(self, mode):
        if mode not in self.MODES:
            return False
        # Load the pose model outside the lock - it takes a moment, and
        # holding the lock would freeze the stream while it loads.
        if mode == "fall_detection" and self.pose_model is None:
            pose_model = YOLO("yolov8n-pose.pt")
            with self.lock:
                if self.pose_model is None:
                    self.pose_model = pose_model
        with self.lock:
            self.mode = mode
        return True

    # ----- detection modes -------------------------------------------------

    def _process_tracking(self, frame):
        results = self.general_model.track(frame, imgsz=self.INFER_SIZE, persist=True, verbose=False)
        annotated = results[0].plot()
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"Objects tracked: {count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated, f"{count} objects tracked"

    def _process_fire_smoke(self, frame):
        results = self.fire_model(frame, imgsz=self.INFER_SIZE, conf=0.55, verbose=False)
        annotated = results[0].plot()
        fire_detected = results[0].boxes is not None and len(results[0].boxes) > 0
        label = "FIRE/SMOKE DETECTED" if fire_detected else "Zone clear"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated, label

    def _process_fall_detection(self, frame):
        results = self.pose_model(frame, imgsz=self.INFER_SIZE, verbose=False)
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

    # ----- threads ---------------------------------------------------------

    def _open_camera(self):
        # CAP_DSHOW is the more reliable backend for webcams on Windows;
        # the default backend can silently fail to deliver frames even
        # when isOpened() reports True.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def _grab_loop(self):
        cap = self._open_camera()
        if not cap.isOpened():
            with self.lock:
                self.camera_available = False
                self._info = {
                    "mode": self.mode, "fps": 0.0, "latency_ms": 0,
                    "detail": "No camera available - this feature requires local hardware.",
                }
                self._jpeg_ready.notify_all()
            return

        failures = 0
        while self._running:
            ok, frame = cap.read()  # blocks until the camera delivers the next frame
            if not ok:
                failures += 1
                if failures % 20 == 0:
                    with self.lock:
                        self._info["detail"] = (
                            f"ERROR: webcam not delivering frames ({failures} failed reads) - "
                            "close any other app using the camera and restart the server"
                        )
                time.sleep(0.05)
                continue
            failures = 0
            with self.lock:
                self._frame = frame
                self._frame_id += 1
                self._frame_time = time.time()
                self._frame_ready.notify_all()

        cap.release()

    def _infer_loop(self):
        try:
            self._warm_up()
        except Exception as e:
            with self.lock:
                self._info["detail"] = f"model warm-up failed: {e}"

        last_id = 0
        fps = 0.0
        last_output = time.time()

        while self._running:
            with self.lock:
                self._frame_ready.wait_for(lambda: self._frame_id != last_id or not self._running, timeout=1.0)
                if self._frame_id == last_id:
                    continue
                frame, last_id, captured_at = self._frame, self._frame_id, self._frame_time
                mode = self.mode
                watched = self._viewer_count > 0

            frame = self._resize(frame)
            if watched:
                try:
                    if mode == "tracking":
                        annotated, detail = self._process_tracking(frame)
                    elif mode == "fire_smoke":
                        annotated, detail = self._process_fire_smoke(frame)
                    elif mode == "fall_detection" and self.pose_model is not None:
                        annotated, detail = self._process_fall_detection(frame)
                    else:
                        annotated, detail = frame, "Loading model..."
                except Exception as e:
                    annotated, detail = frame, f"error: {e}"
            else:
                # Nobody watching: cheap pass-through at a low rate, no model.
                annotated, detail = frame, "Idle (no active viewer)"

            jpeg = self._encode(annotated)
            now = time.time()
            dt = now - last_output
            last_output = now
            if dt > 0:
                fps = (1.0 / dt) if fps == 0 else fps * 0.85 + (1.0 / dt) * 0.15

            with self.lock:
                if jpeg is not None:
                    self._jpeg = jpeg
                    self._jpeg_id += 1
                self._info = {
                    "mode": mode,
                    "detail": detail,
                    "fps": round(fps, 1),
                    "latency_ms": int((now - captured_at) * 1000),
                }
                self._jpeg_ready.notify_all()

            if not watched:
                time.sleep(1.0 / self.IDLE_FPS)

    def start(self):
        with self.lock:
            if self._running:
                return
            self._running = True
        threading.Thread(target=self._grab_loop, name="vision-grab", daemon=True).start()
        threading.Thread(target=self._infer_loop, name="vision-infer", daemon=True).start()

    # ----- consumers -------------------------------------------------------

    def get_jpeg(self):
        with self.lock:
            if not self.camera_available:
                return self._placeholder_jpeg
            return self._jpeg

    def get_info(self):
        with self.lock:
            return dict(self._info)

    def generate_mjpeg(self):
        self.add_viewer()
        last_id = -1
        try:
            while True:
                with self.lock:
                    self._jpeg_ready.wait_for(
                        lambda: self._jpeg_id != last_id or not self.camera_available, timeout=2.0
                    )
                    if not self.camera_available:
                        frame = self._placeholder_jpeg
                    else:
                        frame = self._jpeg
                    last_id = self._jpeg_id
                if frame is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                if not self.camera_available:
                    time.sleep(2.0)
        finally:
            self.remove_viewer()
