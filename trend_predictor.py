import numpy as np
from collections import deque

class TrendPredictor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, risk_score):
        self.history.append(risk_score)

    def get_trend(self):
        if len(self.history) < 3:
            return {"trend": "INSUFFICIENT_DATA", "slope": 0.0, "predicted_next": None}

        x = np.arange(len(self.history))
        y = np.array(self.history)

        # simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]

        if slope > 1.5:
            trend = "RISING_FAST"
        elif slope > 0.3:
            trend = "RISING"
        elif slope < -1.5:
            trend = "FALLING_FAST"
        elif slope < -0.3:
            trend = "FALLING"
        else:
            trend = "STABLE"

        predicted_next = round(y[-1] + slope, 1)

        return {"trend": trend, "slope": round(slope, 2), "predicted_next": predicted_next}


if __name__ == "__main__":
    predictor = TrendPredictor(window_size=8)

    # simulate a risk score climbing steadily
    test_scores = [10, 12, 15, 20, 28, 38, 50, 65]

    for score in test_scores:
        predictor.update(score)
        result = predictor.get_trend()
        print(f"Risk score: {score} -> Trend: {result['trend']}, Slope: {result['slope']}, Predicted next: {result['predicted_next']}")
