# AegisAI
### Autonomous Multimodal Emergency Intelligence & Disaster Response Platform

A real-time AI system for emergency detection, risk prediction, and disaster response - built in phases, starting from core computer vision and scaling toward a full multimodal intelligence platform (sensor fusion, predictive risk modeling, route optimization, and LLM-assisted decision support).

Inspired by how real-world smart-city and campus command centers monitor and respond to emergencies - not just detecting one event type, but reasoning across multiple data sources in real time.

---

## Current status: Phase 2 of 11

- [x] **Phase 1 - Real-time vision pipeline**
  Live object detection (YOLOv8n) and multi-object tracking on webcam feed, running in real time on CPU.

- [x] **Phase 2 - Crowd analytics & fall detection**
  Person-only detection with live crowd counting, and pose-based fall detection using keypoint confidence filtering to avoid false positives from partial-body visibility.

- [ ] Phase 3 - Fire/smoke detection *(in progress)*
- [ ] Phase 4 - IoT sensor fusion
- [ ] Phase 5 - Predictive risk & anomaly detection
- [ ] Phase 6 - Route optimization for evacuation
- [ ] Phase 7 - Drone-based aerial intelligence
- [ ] Phase 8 - NLP for emergency reports/calls
- [ ] Phase 9 - LLM-assisted command center
- [ ] Phase 10 - Digital twin & scenario simulation
- [ ] Phase 11 - Multi-agent orchestration, MLOps & security hardening

---

## Tech stack (so far)

- **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
- **Language:** Python 3.11
- **Runtime:** CPU-only inference (no GPU required for current phases)

---

## Notable engineering finding

Pose-based fall detection using shoulder/hip keypoints only works reliably when the full body is visible and correctly framed - a laptop webcam mounted at desk height frequently fails to capture hip keypoints with sufficient confidence, regardless of whether the person is sitting or lying down. This was empirically validated by logging per-keypoint confidence scores across different distances and poses. Real deployments should use ceiling- or wall-mounted cameras for reliable fall detection - a constraint reflected in this project planned camera architecture for later phases.

---

## Why this project

Built as a hands-on exploration of multimodal AI systems design - going beyond single-model computer vision projects into sensor fusion, predictive modeling, and decision-support architecture, the kind of system used in real emergency response and smart-city platforms.
