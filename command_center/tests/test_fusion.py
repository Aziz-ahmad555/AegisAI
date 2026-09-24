"""Sensor fusion scoring, risk bands, anomaly detection and trend prediction."""
import pytest

from aegis_core.anomaly_detector import AnomalyDetector
from aegis_core.fusion_engine import classify_risk, compute_risk_score
from aegis_core.sensor_simulator import SensorSimulator
from aegis_core.trend_predictor import TrendPredictor

BASELINE = dict(temp=24, smoke_level=5, gas_level=10)


def test_baseline_scores_zero_and_everything_maxed_caps_at_100():
    assert compute_risk_score(0, 0, **BASELINE) == 0.0
    assert compute_risk_score(1, 1, temp=200, smoke_level=500, gas_level=500) == 100.0


def test_camera_alone_is_capped_below_high():
    # Design principle from Phase 4: one modality alone can't reach HIGH.
    score = compute_risk_score(1.0, 1.0, **BASELINE)
    assert score == 40.0 and classify_risk(score) == "ELEVATED"


def test_sensors_alone_are_capped_below_critical():
    score = compute_risk_score(0, 0, temp=200, smoke_level=500, gas_level=500)
    assert score == 60.0 and classify_risk(score) == "HIGH"


def test_strong_agreement_between_modalities_reaches_critical():
    # camera 0.8 -> 32, temp 55 -> 17.22, smoke 70 -> 17.33, gas 40 -> 12  = 78.56
    score = compute_risk_score(0.8, 0.3, temp=55, smoke_level=70, gas_level=40)
    assert score == 78.6 and classify_risk(score) == "CRITICAL"


def test_moderate_agreement_is_high_not_critical():
    score = compute_risk_score(0.7, 0.5, temp=45, smoke_level=50, gas_level=35)
    assert score == 61.7 and classify_risk(score) == "HIGH"


def test_below_baseline_readings_do_not_go_negative():
    assert compute_risk_score(0, 0, temp=0, smoke_level=0, gas_level=0) == 0.0


@pytest.mark.parametrize("score, level", [
    (0, "NORMAL"), (19.9, "NORMAL"), (20, "ELEVATED"), (44.9, "ELEVATED"),
    (45, "HIGH"), (69.9, "HIGH"), (70, "CRITICAL"), (100, "CRITICAL"),
])
def test_risk_band_boundaries(score, level):
    assert classify_risk(score) == level


def _stable(detector, n=10):
    for i in range(n):
        detector.update_and_check({"temperature": 24.0 + (i % 2) * 0.1, "smoke_level": 5.0, "gas_level": 10.0})


def test_anomaly_needs_history_before_flagging():
    d = AnomalyDetector()
    result = d.update_and_check({"temperature": 90.0, "smoke_level": 5.0, "gas_level": 10.0})
    assert not result["temperature"]["is_anomaly"]


def test_spike_is_flagged_after_stable_history():
    d = AnomalyDetector()
    _stable(d)
    result = d.update_and_check({"temperature": 55.0, "smoke_level": 5.0, "gas_level": 10.0})
    assert result["temperature"]["is_anomaly"]
    assert not result["smoke_level"]["is_anomaly"]


def test_tiny_change_on_ultra_stable_baseline_is_not_anomalous():
    # High z-score but below the minimum absolute delta -> not a real anomaly.
    d = AnomalyDetector()
    _stable(d)
    result = d.update_and_check({"temperature": 24.6, "smoke_level": 5.0, "gas_level": 10.0})
    assert result["temperature"]["z_score"] > 2.5
    assert not result["temperature"]["is_anomaly"]


@pytest.mark.parametrize("series, trend", [
    ([10, 10, 10, 10], "STABLE"),
    ([10, 11, 12, 13], "RISING"),
    ([10, 20, 30, 40], "RISING_FAST"),
    ([40, 39, 38, 37], "FALLING"),
    ([40, 30, 20, 10], "FALLING_FAST"),
])
def test_trend_classification(series, trend):
    t = TrendPredictor(window_size=8)
    for x in series:
        t.update(x)
    assert t.get_trend()["trend"] == trend


def test_trend_needs_three_points_and_predicts_next():
    t = TrendPredictor()
    t.update(10)
    t.update(20)
    assert t.get_trend()["trend"] == "INSUFFICIENT_DATA"
    t.update(30)
    assert t.get_trend()["predicted_next"] == 40.0


def test_simulator_ramps_gradually_toward_event_and_back():
    sim = SensorSimulator()
    sim.cooldown_until = float("inf")        # no random events
    sim.trigger_event_manually()
    first = sim.read()["temperature"]
    assert 24 < first < 40                   # ramps, doesn't jump to 58
    for _ in range(200):
        sim.read()
    assert not sim.event_active              # event ends on its own
    for _ in range(200):
        sim.read()
    assert abs(sim.read()["temperature"] - 24) < 2
