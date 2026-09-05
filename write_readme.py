content = """# AegisAI
### Autonomous Multimodal Emergency Intelligence & Disaster Response Platform

A real-time AI system for emergency detection, risk prediction, and disaster response - built in phases, starting from core computer vision and scaling toward a full multimodal intelligence platform (sensor fusion, predictive risk modeling, route optimization, and LLM-assisted decision support).

Inspired by how real-world smart-city and campus command centers monitor and respond to emergencies - not just detecting one event type, but reasoning across multiple data sources in real time.

---

## Current status: Phase 4 of 11

- [x] Phase 1 - Real-time vision pipeline: Live object detection (YOLOv8n) and multi-object tracking on webcam feed, running in real time on CPU.
- [x] Phase 2 - Crowd analytics and fall detection: Person-only detection with live crowd counting, and pose-based fall detection using keypoint confidence filtering.
- [x] Phase 3 - Fire/smoke detection: Custom-trained YOLOv8n model (30 epochs, CPU-only). mAP50: 0.576 (Fire: 0.578, Smoke: 0.575). Documented limitation: reduced reliability in low-light conditions.
- [ ] Phase 4 - IoT sensor fusion (in progress)
- [ ] Phase 5 - Predictive risk and anomaly detection
- [ ] Phase 6 - Route optimization for evacuation
- [ ] Phase 7 - Drone-based aerial intelligence
- [ ] Phase 8 - NLP for emergency reports/calls
- [ ] Phase 9 - LLM-assisted command center
- [ ] Phase 10 - Digital twin and scenario simulation
- [ ] Phase 11 - Multi-agent orchestration, MLOps and security hardening

---

## Tech stack (so far)

- Computer Vision: YOLOv8 (Ultralytics), OpenCV
- Custom Model Training: Roboflow (dataset), Ultralytics CLI training
- Language: Python 3.11
- Runtime: CPU-only inference and training (no GPU required)

---

## Notable engineering findings

- Pose-based fall detection using shoulder/hip keypoints only works reliably when the full body is visible and correctly framed. Validated empirically by logging per-keypoint confidence scores.
- Custom fire/smoke detection shows reduced confidence in low-light conditions due to a daylight-skewed training dataset - a common day/night domain gap in computer vision.

---

## Why this project

Built as a hands-on exploration of multimodal AI systems design - going beyond single-model computer vision projects into sensor fusion, predictive modeling, and decision-support architecture.
"""

with open("phase1_vision/README.md", "w", encoding="utf-8") as f:
    f.write(content)

print("README written successfully")
