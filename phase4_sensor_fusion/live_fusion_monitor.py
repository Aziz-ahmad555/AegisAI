import cv2
import time
from ultralytics import YOLO
from sensor_simulator import SensorSimulator
from fusion_engine import compute_risk_score, classify_risk

model = YOLO("fire_smoke_model.pt")
sensors = SensorSimulator()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Starting AegisAI Phase 4 live fusion monitor. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Camera detection ---
    results = model(frame, imgsz=416, conf=0.4, verbose=False)
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

    # --- Sensor reading ---
    reading = sensors.read()

    # --- Fusion ---
    risk_score = compute_risk_score(
        camera_fire_conf=fire_conf,
        camera_smoke_conf=smoke_conf,
        temp=reading["temperature"],
        smoke_level=reading["smoke_level"],
        gas_level=reading["gas_level"],
    )
    risk_level = classify_risk(risk_score)

    # --- Overlay info on video ---
    color = (0, 255, 0) if risk_level == "NORMAL" else \
            (0, 255, 255) if risk_level == "ELEVATED" else \
            (0, 128, 255) if risk_level == "HIGH" else (0, 0, 255)

    cv2.putText(annotated, f"Risk: {risk_score} ({risk_level})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(annotated, f"Temp: {reading['temperature']}C Smoke: {reading['smoke_level']} Gas: {reading['gas_level']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("AegisAI - Phase 4 Multimodal Fusion", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
