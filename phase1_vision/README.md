# AegisAI
### Autonomous Multimodal Emergency Intelligence & Disaster Response Platform

A real-time AI system for emergency detection, risk prediction, and disaster response - built in phases, starting from core computer vision and scaling toward a full multimodal intelligence platform (sensor fusion, predictive risk modeling, route optimization, and LLM-assisted decision support).

Inspired by how real-world smart-city and campus command centers monitor and respond to emergencies - not just detecting one event type, but reasoning across multiple data sources in real time.

---

## Current status: Phase 8 of 11

- [x] **Phase 1 - Real-time vision pipeline**
  Live object detection (YOLOv8n) and multi-object tracking on webcam feed, running in real time on CPU.

- [x] **Phase 2 - Crowd analytics and fall detection**
  Person-only detection with live crowd counting, and pose-based fall detection using keypoint confidence filtering to avoid false positives from partial-body visibility.
- [x] **Phase 3 - Fire/smoke detection**
  Custom-trained YOLOv8n model (30 epochs, CPU-only) on a public fire/smoke dataset. mAP50: 0.576 (Fire: 0.578, Smoke: 0.575). Documented limitation: reduced reliability in low-light conditions due to limited nighttime training data.

- [x] **Phase 4 - IoT sensor fusion**
  Simulated temperature/smoke/gas sensor stream combined with live camera fire/smoke detection into a single fused risk score (NORMAL / ELEVATED / HIGH / CRITICAL). Demonstrates multimodal reasoning: a single signal raises moderate concern, but agreement between both modalities produces a much higher, more confident risk assessment.

- [x] **Phase 5 - Predictive risk and anomaly detection**
  Rolling statistical anomaly detector (z-score based) flags unusual sensor readings relative to recent history. Linear trend predictor classifies risk trajectory (STABLE/RISING/RISING_FAST/FALLING) and forecasts the next value. Combined into a full live monitor alongside camera detection and sensor fusion. Sensor simulator rewritten with gradual value ramping and cooldown periods to model realistic physical behavior instead of instant jumps.
- [x] **Phase 6 - Route optimization for evacuation**
  Graph-based building model (rooms, corridors, exits) with Dijkstra shortest-path evacuation routing via NetworkX. Validated dynamic rerouting when paths are blocked, including correct detection of fully isolated/unreachable rooms. Integrated with Phase 3's fire detection model: camera-detected fire in a monitored zone automatically blocks the corresponding graph edges and triggers live rerouting - demonstrating a working end-to-end pipeline from computer vision to decision-making.
- [x] **Phase 7 - Drone-based aerial intelligence**
  Custom-trained YOLOv8n model (30 epochs, CPU-only) on the SARD (Search and Rescue Drone) dataset - 1,980 aerial images of people in distress-relevant poses (Running, Walking, laying_down, seated, stands). mAP50: 0.572, Precision: 0.784, Recall: 0.512. Tested on static aerial test images rather than live drone footage, since no physical drone hardware was available - a deliberate, honestly-scoped simulation of aerial search-and-rescue analysis. Distinguishing laying_down/seated poses from active movement directly supports identifying potentially injured or stranded individuals in disaster imagery.
- [x] **Phase 8 - NLP for emergency reports/calls**
  Hybrid NLP pipeline combining spaCy's pretrained entity recognition with custom rule-based classification for emergency-specific concepts spaCy doesn't natively understand: event type (fire vs. possible-fire-smell distinction, medical, structural, trapped), severity assessment (including detection of reporter-downplayed language), people counts, and common indoor location vocabulary (kitchen, cafeteria, laboratory, etc.) that general-purpose NER misses. Interactive CLI analyzer included. Iteratively debugged real issues: an operator-precedence bug in number parsing, keyword collisions between event categories, and substring double-matching in location extraction.
- [ ] Phase 9 - LLM-assisted command center
- [ ] Phase 10 - Digital twin and scenario simulation
- [ ] Phase 11 - Multi-agent orchestration, MLOps and security hardening

---

## Tech stack

- **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
- **Custom Model Training:** Roboflow (dataset), Ultralytics CLI
- **Sensor Simulation / Fusion Logic:** Python
- **Graph Algorithms / Route Optimization:** NetworkX, Matplotlib
- **NLP:** spaCy (pretrained NER) + custom rule-based classification
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

### Phase 6 - Route optimization (phase6_route_optimization/)
Requires the trained model from Phase 3 copied in as fire_smoke_model.pt
pip install networkx matplotlib ultralytics opencv-python
python building_graph.py
python route_finder.py
python integrated_monitor.py

### Phase 7 - Drone aerial intelligence (phase7_drone_intelligence/)
Model weights are not included in this repo.
pip install ultralytics roboflow
python download_dataset.py
yolo task=detect mode=train model=yolov8n.pt data=Sard-4/data.yaml epochs=30 imgsz=416 batch=8 device=cpu
python test_on_images.py

### Phase 8 - NLP emergency reports (phase8_nlp_reports/)
pip install spacy
python -m spacy download en_core_web_sm
python emergency_nlp.py
python interactive_report_analyzer.py

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
  phase5_prediction/      # Anomaly detection, trend prediction, full live monitor
  phase6_route_optimization/  # Graph-based evacuation routing, integrated with fire detection
  phase7_drone_intelligence/  # Aerial search-and-rescue pose detection (SARD dataset)
  phase8_nlp_reports/     # Hybrid NLP pipeline for emergency report/call analysis

---

## Why this project

Built as a hands-on exploration of multimodal AI systems design - going beyond single-model computer vision projects into sensor fusion, predictive modeling, and decision-support architecture.











