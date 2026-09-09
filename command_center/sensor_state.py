import threading
import time
from collections import deque
from sensor_simulator import SensorSimulator
from fusion_engine import compute_risk_score, classify_risk
from anomaly_detector import AnomalyDetector
from trend_predictor import TrendPredictor


class SensorFusionState:
    """
    Live, continuously-updating multimodal sensor fusion state.
    Combines simulated IoT sensor readings with a (currently zero, since no
    live camera feed is wired into this section yet) camera confidence input
    into a single fused risk score, with rolling anomaly detection and trend
    forecasting layered on top - directly mirroring Phases 4 and 5.
    """

    def __init__(self, history_length=30):
        self.simulator = SensorSimulator()
        self.anomaly_detector = AnomalyDetector(window_size=15, z_threshold=2.5)
        self.trend_predictor = TrendPredictor(window_size=history_length)
        self.lock = threading.RLock()

        self.latest_reading = {"temperature": 24.0, "smoke_level": 5.0, "gas_level": 10.0}
        self.latest_anomaly = False
        self.latest_risk_score = 0.0
        self.latest_risk_level = "NORMAL"
        self.latest_trend = {"trend": "STABLE", "slope": 0.0, "predicted_next": 0.0}
        self.risk_history = deque(maxlen=history_length)
        self.anomaly_hold_until = 0
        self.ANOMALY_HOLD_SECONDS = 4

        self._running = False

    def update_once(self, camera_fire_conf=0.0, camera_smoke_conf=0.0):
        with self.lock:
            # Disable the simulator's built-in random auto-trigger for this
            # dashboard - a portfolio demo should behave deterministically so
            # every viewing session is controllable and predictable via the
            # manual "Simulate Sensor Event" button, rather than surprising
            # a viewer with an unexplained spike they didn't cause.
            self.simulator.cooldown_until = time.time() + 999999
            reading = self.simulator.read()
            self.latest_reading = reading

            anomaly_result = self.anomaly_detector.update_and_check(reading)
            any_anomaly_now = any(info["is_anomaly"] for info in anomaly_result.values())
            if any_anomaly_now:
                self.anomaly_hold_until = time.time() + self.ANOMALY_HOLD_SECONDS
            self.latest_anomaly = time.time() < self.anomaly_hold_until

            risk_score = compute_risk_score(
                camera_fire_conf=camera_fire_conf,
                camera_smoke_conf=camera_smoke_conf,
                temp=reading["temperature"],
                smoke_level=reading["smoke_level"],
                gas_level=reading["gas_level"],
            )
            self.latest_risk_score = risk_score
            self.latest_risk_level = classify_risk(risk_score)

            self.trend_predictor.update(risk_score)
            self.latest_trend = self.trend_predictor.get_trend()

            self.risk_history.append(risk_score)

    def trigger_event(self):
        with self.lock:
            self.simulator.trigger_event_manually()

    def get_snapshot(self):
        with self.lock:
            return {
                "reading": self.latest_reading,
                "anomaly": self.latest_anomaly,
                "risk_score": self.latest_risk_score,
                "risk_level": self.latest_risk_level,
                "trend": self.latest_trend,
                "risk_history": list(self.risk_history),
            }

    def start_background_loop(self, interval_seconds=1.0):
        if self._running:
            return
        self._running = True

        def loop():
            while True:
                self.update_once()
                time.sleep(interval_seconds)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
