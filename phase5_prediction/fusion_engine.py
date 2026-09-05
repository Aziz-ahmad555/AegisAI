def compute_risk_score(camera_fire_conf, camera_smoke_conf, temp, smoke_level, gas_level):
    """
    Combines camera detection confidence with sensor readings into a single
    0-100 risk score. Weights are intentionally simple/tunable - a real system
    would derive these from calibration data, not guesswork.
    """

    # --- Camera contribution (0-40 points) ---
    camera_score = max(camera_fire_conf, camera_smoke_conf) * 40

    # --- Sensor contribution (0-60 points) ---
    # normalize each sensor against a "normal vs alarming" range
    temp_score = min(max((temp - 24) / (60 - 24), 0), 1) * 20
    smoke_score = min(max((smoke_level - 5) / (80 - 5), 0), 1) * 20
    gas_score = min(max((gas_level - 10) / (60 - 10), 0), 1) * 20

    sensor_score = temp_score + smoke_score + gas_score

    total = camera_score + sensor_score

    return round(min(total, 100), 1)


def classify_risk(score):
    if score < 20:
        return "NORMAL"
    elif score < 45:
        return "ELEVATED"
    elif score < 70:
        return "HIGH"
    else:
        return "CRITICAL"


if __name__ == "__main__":
    # quick manual test cases before wiring in real camera + sensors
    test_cases = [
        {"camera_fire_conf": 0.0, "camera_smoke_conf": 0.0, "temp": 24, "smoke_level": 5, "gas_level": 10},
        {"camera_fire_conf": 0.6, "camera_smoke_conf": 0.0, "temp": 25, "smoke_level": 6, "gas_level": 11},
        {"camera_fire_conf": 0.7, "camera_smoke_conf": 0.5, "temp": 45, "smoke_level": 50, "gas_level": 35},
        {"camera_fire_conf": 0.0, "camera_smoke_conf": 0.0, "temp": 55, "smoke_level": 70, "gas_level": 40},
    ]

    for i, case in enumerate(test_cases, 1):
        score = compute_risk_score(**case)
        level = classify_risk(score)
        print(f"Case {i}: {case}")
        print(f"  -> Risk Score: {score}  |  Level: {level}\n")
