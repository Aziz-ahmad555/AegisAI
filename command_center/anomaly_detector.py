import numpy as np
from collections import deque


class AnomalyDetector:
    def __init__(self, window_size=20, z_threshold=2.5, min_absolute_delta=None):
        """
        window_size: how many recent readings to consider "normal history"
        z_threshold: how many standard deviations away counts as anomalous
        min_absolute_delta: per-key minimum absolute change from the mean
            required (in addition to the z-score) before flagging an anomaly.
            This prevents hypersensitivity when the baseline is extremely
            stable - a tiny absolute fluctuation can still produce a large
            z-score purely because the recent variance is small, which is
            statistically correct but not a meaningful real-world anomaly.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_absolute_delta = min_absolute_delta or {
            "temperature": 2.0,
            "smoke_level": 3.0,
            "gas_level": 3.0,
        }
        self.history = {
            "temperature": deque(maxlen=window_size),
            "smoke_level": deque(maxlen=window_size),
            "gas_level": deque(maxlen=window_size),
        }

    def update_and_check(self, reading):
        anomalies = {}

        for key, value in reading.items():
            hist = self.history[key]

            if len(hist) >= 5:
                mean = np.mean(hist)
                std = np.std(hist)
                abs_delta = abs(value - mean)
                min_delta = self.min_absolute_delta.get(key, 0)

                if std > 0:
                    z_score = abs_delta / std
                    is_anomaly = (z_score > self.z_threshold) and (abs_delta > min_delta)
                    anomalies[key] = {
                        "value": value,
                        "z_score": round(z_score, 2),
                        "is_anomaly": is_anomaly,
                    }
                else:
                    anomalies[key] = {"value": value, "z_score": 0.0, "is_anomaly": False}
            else:
                anomalies[key] = {"value": value, "z_score": 0.0, "is_anomaly": False}

            hist.append(value)

        return anomalies


if __name__ == "__main__":
    detector = AnomalyDetector(window_size=10, z_threshold=2.5)

    test_stream = [
        {"temperature": 24.0, "smoke_level": 5.0, "gas_level": 10.0},
        {"temperature": 24.2, "smoke_level": 5.1, "gas_level": 10.2},
        {"temperature": 23.8, "smoke_level": 4.9, "gas_level": 9.8},
        {"temperature": 24.1, "smoke_level": 5.0, "gas_level": 10.1},
        {"temperature": 24.0, "smoke_level": 5.2, "gas_level": 9.9},
        {"temperature": 23.9, "smoke_level": 4.8, "gas_level": 10.0},
        {"temperature": 24.3, "smoke_level": 5.1, "gas_level": 10.3},
        {"temperature": 55.0, "smoke_level": 60.0, "gas_level": 45.0},
        {"temperature": 24.1, "smoke_level": 5.0, "gas_level": 10.0},
    ]

    for i, reading in enumerate(test_stream, 1):
        result = detector.update_and_check(reading)
        print(f"Reading {i}: {reading}")
        for key, info in result.items():
            flag = "  <-- ANOMALY" if info["is_anomaly"] else ""
            print(f"  {key}: value={info['value']}, z={info['z_score']}{flag}")
        print()
