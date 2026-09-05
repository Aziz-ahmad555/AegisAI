# AegisAI
### Autonomous Multimodal Emergency Intelligence & Disaster Response Platform

A real-time AI system for emergency detection, risk prediction, and disaster response - built in phases, starting from core computer vision and scaling toward a full multimodal intelligence platform (sensor fusion, predictive risk modeling, route optimization, and LLM-assisted decision support).

Inspired by how real-world smart-city and campus command centers monitor and respond to emergencies - not just detecting one event type, but reasoning across multiple data sources in real time.

---

## Current status: Phase 4 of 11

- [x] **Phase 1 - Real-time vision pipeline**
  Live object detection (YOLOv8n) and multi-object tracking on webcam feed, running in real time on CPU.

- [x] **Phase 2 - Crowd analytics and fall detection**
  Person-only detection with live crowd counting, and pose-based fall detection using keypoint confidence filtering to avoid false positives from partial-body visibility.
- [x] **Phase 3 - Fire/smoke detection**
  Custom-trained YOLOv8n model (30 epochs, CPU-only) on a public fire/smoke dataset. mAP50: 0.576 (Fire: 0.578, Smoke: 0.575). Documented limitation: reduced reliability in low-light conditions due to limited nighttime training data.

- [x] **Phase 4 - IoT sensor fusion**
  Simulated temperature/smoke/gas sensor stream combined with live camera fire/smoke detection into a single fused risk score (NORMAL / ELEVATED / HIGH / CRITICAL). Demonstrates multimodal reasoning: a single signal raises moderate concern, but agreement between both modalities produces a much higher, more confident risk assessment.

- [ ] Phase 5 - Predictive risk and anomaly detection
- [ ] Phase 6 - Route optimization for evacuation
- [ ] Phase 7 - Drone-based aerial intelligence
- [ ] Phase 8 - NLP for emergency reports/calls
- [ ] Phase 9 - LLM-assisted command center
- [ ] Phase 10 - Digital twin and scenario simulation
- [ ] Phase 11 - Multi-agent orchestration, MLOps and security hardening

---

## Tech stack

- **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
- **Custom Model Training:** Roboflow (dataset), Ultralytics CLI
- **Sensor Simulation / Fusion Logic:** Python
- **Language:** Python 3.11
- **Runtime:** CPU-only inference and training (no GPU required)

---

## How to run this project

Each phase is self-contained with its own virtual environment.

### Phase 1 - Vision pipeline (phase1_vision/)
pip install ultralytics opencv-python
python detect_webcam.py
python crowd_count.py
python fall_detection.py

### Phase 3 - Fire/smoke detection (phase3_fire_smoke/)
Model weights are not included in this repo.
pip install ultralytics roboflow
python download_dataset.py
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=30 imgsz=416 batch=8 device=cpu

### Phase 4 - Sensor fusion (phase4_sensor_fusion/)
Requires the trained model from Phase 3 copied in as fire_smoke_model.pt
pip install ultralytics opencv-python
python sensor_simulator.py
python fusion_engine.py
python live_fusion_monitor.py

---

## Model Weights

Trained model weights (.pt files) are intentionally excluded from this repository. This is standard practice for ML projects - it keeps the repo lightweight and forces a reproducible training pipeline rather than relying on a committed binary. Training takes roughly 1-1.5 hours on a CPU-only laptop (tested on Intel i5 8th gen, 16GB RAM).

---

## Notable engineering findings

- **Fall detection camera-angle limitation:** Pose-based fall detection only works reliably when the full body is visible and correctly framed. A laptop webcam at desk height frequently fails to capture hip keypoints with sufficient confidence. Validated empirically by logging per-keypoint confidence scores across distances and poses.

- **Fire/smoke low-light domain gap:** The custom-trained detector shows reduced confidence in low-light conditions since the training dataset skews toward daylight scenes - a known day/night domain gap in computer vision.

- **Multimodal fusion reduces false confidence:** Camera-only detections are capped at moderate risk scores by design. Only when sensor readings and camera detection agree does the fused risk score reach HIGH/CRITICAL - demonstrating why real emergency systems combine multiple sensor types.

---

## Project structure

AegisAI/
  phase1_vision/          # Detection, tracking, crowd counting, fall detection
  phase3_fire_smoke/      # Custom fire/smoke model training pipeline
  phase4_sensor_fusion/   # Sensor simulation + multimodal risk fusion engine

---

## Why this project

Built as a hands-on exploration of multimodal AI systems design - going beyond single-model computer vision projects into sensor fusion, predictive modeling, and decision-support architecture.

