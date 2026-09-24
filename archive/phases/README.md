# Archived phase snapshots

These folders are the **historical development snapshots** of AegisAI, one per build phase (Phase 1 vision through Phase 11 multi-agent / MLOps), kept exactly as they were when each phase was built and tested.

- They are **not maintained**. Code here is not kept in sync with the live system, and several modules are earlier copies of what now lives in `aegis_core`.
- The **live code** is:
  - [`aegis_core/`](../../aegis_core): the shared domain logic (digital twin, sensor fusion, anomaly/trend detection, NLP, agents, event bus, vision pipeline);
  - [`command_center/`](../../command_center): the web app that runs it.
- They stay in the repo because they document how the system was built. The build history and each phase's run instructions are [below](#running-an-individual-phase).
- Local-only content that moved with them (per-phase `venv/`, datasets, training `runs/`, model weights) is git-ignored. Windows venvs contain absolute paths, so a moved phase venv's `activate`/`pip.exe` may point at the old location. Recreate it or use `venv\Scripts\python.exe -m pip` if you need to run a phase again.
- Model training pipelines (to reproduce weights) live here: fire/smoke in `phase3_fire_smoke/`, aerial search-and-rescue in `phase7_drone_intelligence/`.

---

# Build history and per-phase instructions

*Moved here from the main README; kept as written when each phase was built. Paths are relative to this folder.*

## Phase-by-phase summary

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
- [x] **Phase 9 - LLM-assisted command center**
  Claude (Anthropic API) wired with tool-use/function-calling to answer natural-language questions about live system state (active incidents, risk score, evacuation status, parsed reports) grounded in real data rather than hallucination. Includes a rule-based offline fallback that activates automatically on API errors (insufficient credits, no connection, missing key) so the tool degrades gracefully instead of crashing - a deliberate resilience pattern for a system with an external paid dependency.
- [x] **Phase 10 - Digital twin and scenario simulation**
  Live web dashboard (Flask + Flask-SocketIO) rendering the building graph as an interactive SVG, updated in real time via WebSocket. Operators can trigger/clear simulated fires in any zone and query evacuation routes from any room, with the graph, risk score, and event log all updating live. Fixed a critical evacuation-logic flaw during development: originally, a room catching fire blocked its own only exit, incorrectly reporting occupants as fully trapped even when their own door was still physically usable. Redesigned the routing logic to distinguish three cases - a clean route avoiding all hazards, a route that must pass through the occupant's own hazard zone (flagged with a visible warning, since there is no alternate physical option), and genuine full isolation (correctly reported as requiring rescue). Also fixed a critical deadlock bug during backend development: a non-reentrant threading.Lock() caused the state snapshot method to hang indefinitely when it called another locked method from within an already-locked block - resolved by switching to threading.RLock().
- [x] **Phase 11 - Multi-agent orchestration, MLOps and security hardening**
  Multi-agent system with three specialist agents (Fire, Medical, Route) and an LLM-based Decision Agent coordinator that selectively consults only the agents relevant to each operator query, rather than always querying everything - includes a keyword-based offline routing fallback so selective consultation still works without API access. MLOps practices documented (model versioning, lightweight registry via recorded metrics, manually-identified failure modes, honest scoping of what a production deployment would still need). Added session-based authentication to the Phase 10 dashboard, protecting both HTTP routes and WebSocket actions, with credentials configurable via environment variables rather than hardcoded.

---

## Tech stack

- **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
- **Custom Model Training:** Roboflow (dataset), Ultralytics CLI
- **Sensor Simulation / Fusion Logic:** Python
- **Graph Algorithms / Route Optimization:** NetworkX, Matplotlib
- **NLP:** spaCy (pretrained NER) + custom rule-based classification
- **LLM Integration:** Anthropic Claude API (tool-use/function-calling)
- **Web Dashboard:** Flask, Flask-SocketIO (real-time WebSocket updates), vanilla JS/SVG frontend
- **Multi-Agent Architecture:** Anthropic Claude API with specialist agent delegation
- **Security:** Session-based authentication (Flask sessions), environment-variable credentials
- **Language:** Python 3.11
- **Runtime:** CPU-only inference and training (no GPU required)

---

## Running an individual phase

The per-phase folders are historical snapshots, now under `archive/phases/` (see its README). Each is self-contained with its own virtual environment; paths below are relative to `archive/phases/`. The live code is `aegis_core/` + `command_center/`.

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

### Phase 9 - LLM command center (phase9_llm_command_center/)
Requires an Anthropic API key set as environment variable ANTHROPIC_API_KEY. Falls back to an offline rule-based summary automatically if the key is missing or the API is unavailable.
pip install anthropic
python command_center.py

### Phase 10 - Digital twin dashboard (phase10_digital_twin/)
pip install flask flask-socketio networkx
python server.py
Then open http://127.0.0.1:5000 in a browser. Login required (default operator / aegisai2026, configurable via AEGISAI_USERNAME / AEGISAI_PASSWORD env vars).

### Phase 11 - Multi-agent coordinator (phase11_multiagent_mlops_security/)
Requires an Anthropic API key for full LLM-based routing; falls back to keyword-based offline routing otherwise.
pip install anthropic
python agents.py
python coordinator.py
See `archive/phases/phase11_multiagent_mlops_security/MLOPS.md` for MLOps practices documentation.

---
