import cv2
import time
from ultralytics import YOLO
from sensor_simulator import SensorSimulator
from fusion_engine import compute_risk_score, classify_risk
from anomaly_detector import AnomalyDetector
from trend_predictor import TrendPredictor

model = YOLO("fire_smoke_model.pt")
sensors = SensorSimulator()
anomaly_detector = AnomalyDetector(window_size=15, z_threshold=2.5)
trend_predictor = TrendPredictor(window_size=10)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Starting AegisAI Phase 5 full monitor.")
print("Press 'e' to manually trigger a test event. Press 'q' to quit.\n")

last_sensor_update = 0
sensor_update_interval = 1.0

anomaly_hold_until = 0
ANOMALY_HOLD_SECONDS = 4  # keep showing the warning for a few seconds after it clears

reading = sensors.read()
anomaly_result = anomaly_detector.update_and_check(reading)
risk_score = 0
risk_level = "NORMAL"
trend_result = {"trend": "STABLE", "predicted_next": 0}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=416, conf=0.55, verbose=False)
    annotated = results[0].plot()

    fire_conf = 0.0
    smoke_conf = 0.0
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            if class_name == "Fire":
                fire_conf = max(fire_conf, conf)
            elif class_name == "Smoke":
                smoke_conf = max(smoke_conf, conf)

    now = time.time()
    if now - last_sensor_update >= sensor_update_interval:
        last_sensor_update = now

        reading = sensors.read()
        anomaly_result = anomaly_detector.update_and_check(reading)
        any_anomaly_now = any(info["is_anomaly"] for info in anomaly_result.values())

        if any_anomaly_now:
            anomaly_hold_until = now + ANOMALY_HOLD_SECONDS

        risk_score = compute_risk_score(
            camera_fire_conf=fire_conf,
            camera_smoke_conf=smoke_conf,
            temp=reading["temperature"],
            smoke_level=reading["smoke_level"],
            gas_level=reading["gas_level"],
        )
        risk_level = classify_risk(risk_score)

        trend_predictor.update(risk_score)
        trend_result = trend_predictor.get_trend()

    show_anomaly_warning = time.time() < anomaly_hold_until

    color = (0, 255, 0) if risk_level == "NORMAL" else \
            (0, 255, 255) if risk_level == "ELEVATED" else \
            (0, 128, 255) if risk_level == "HIGH" else (0, 0, 255)

    cv2.putText(annotated, f"Risk: {risk_score} ({risk_level})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(annotated, f"Temp: {reading['temperature']}C Smoke: {reading['smoke_level']} Gas: {reading['gas_level']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"Trend: {trend_result['trend']} (predicted next: {trend_result['predicted_next']})",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if show_anomaly_warning:
        cv2.putText(annotated, "ANOMALY DETECTED IN SENSOR DATA", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(annotated, "Press 'e' to trigger test event, 'q' to quit", (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("AegisAI - Phase 5 Full Monitor", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        sensors.trigger_event_manually()

cap.release()
cv2.destroyAllWindows()
