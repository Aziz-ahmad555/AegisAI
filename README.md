# AegisAI
### Multimodal emergency intelligence for buildings - detection, fusion, evacuation routing and an agent-driven command chat

[![CI](https://github.com/Aziz-ahmad555/AegisAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Aziz-ahmad555/AegisAI/actions/workflows/ci.yml)

**Live demo: https://aegisai-9aki.onrender.com**
*Hosted on Render's free tier - if it has been idle, the first load can take up to a minute while it wakes up. Demo login available on request. The hosted version has no camera, so Live Vision is local-only; everything else, including the guided scenario, works.*

AegisAI is a command center that reasons across several signals at once, the way a real campus or smart-building control room would: camera fire/smoke detection, IoT-style sensor telemetry, free-text caller reports and a live digital twin of the building. Evidence is fused into a risk picture, the system *proposes* actions (a human confirms them), evacuation routes recompute around hazards, and a decision agent answers operator questions from live data.

![Guided scenario: sensors rise, the camera confirms smoke, a caller reports fire, the fire is confirmed, routes recompute and the agents brief the operator](docs/images/scenario.gif)

*The guided demo scenario (every step is labelled SIMULATED). Recorded in offline mode - no LLM key needed.*

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [What's real and what's simulated](#whats-real-and-whats-simulated)
- [Model metrics](#model-metrics)
- [Run it locally](#run-it-locally)
- [Deploy it (free)](#deploy-it-free)
- [Configuration](#configuration)
- [Security](#security)
- [Testing and CI](#testing-and-ci)
- [Known limitations](#known-limitations)
- [Engineering findings](#engineering-findings)
- [Project structure](#project-structure)

---

## What it does

| | |
|---|---|
| **Operating picture** - one threat level for the building (always on screen), active incidents first, and a live incident timeline across every module. | ![Overview during an incident](docs/images/overview-scenario.png) |
| **Digital twin** - the building as a graph; declared fires block connections and Dijkstra routing recomputes an exit for every room, flagging rooms that must leave through their own hazard zone and rooms that are fully isolated. | ![Digital twin with a fire and an isolated room](docs/images/twin-fire.png) |
| **Command chat** - a decision agent (Groq or Claude, with an offline fallback) consults Fire, Medical and Route agents that read live state, and answers with what the data shows - unknowns stay unknown, caller text is treated as untrusted. | ![Command chat with an agent briefing](docs/images/chat-briefing.png) |
| **Sensor fusion** - temperature, smoke and gas readings fused with camera confidence into one 0-100 risk score, with z-score anomaly detection and trend forecasting. Sensors alone top out at HIGH; CRITICAL needs the camera to agree. | ![Sensor fusion page](docs/images/sensors.png) |
| **Report analyzer** - spaCy entities plus domain rules extract event type, severity (incl. downplayed language), people counts and locations; a fire report that names a zone proposes a fire declaration. | ![Report analyzer](docs/images/report-analyzer.png) |

Also: **Live Vision** (local only - YOLOv8 tracking, the custom fire/smoke model and pose-based fall detection on a webcam, ~120 ms camera-to-screen) and **Aerial Search** (pose classification on search-and-rescue drone imagery).

**Human in the loop.** Automated evidence - a camera detection, CRITICAL sensors, a caller's report - only ever *proposes* "declare fire?". An operator confirms or dismisses it; the guided scenario auto-confirms after a visible countdown, and says so.

---

## Architecture

```mermaid
flowchart LR
    subgraph Inputs
        CAM["Camera<br/>YOLOv8: tracking, fire/smoke, pose"]
        SEN["Sensors<br/>temp / smoke / gas (simulated)"]
        REP["Caller reports<br/>spaCy + rules"]
    end

    CAM -->|fire/smoke confidence| FUS["Sensor fusion<br/>risk score, anomaly, trend"]
    SEN --> FUS
    FUS --> SYS
    CAM -.->|detection| SYS
    REP --> SYS

    SYS["AegisSystem<br/>event bus + incident timeline<br/>proposed actions"] -->|"operator confirms"| TWIN["Digital twin<br/>zones, fires, Dijkstra routes"]

    TWIN --> AGENTS["Fire / Medical / Route agents<br/>(read live state)"]
    SYS --> AGENTS
    AGENTS --> COORD["Decision agent<br/>Groq | Claude | offline routing"]

    SYS -->|Socket.IO| UI["Command Center UI<br/>Flask, server-sanitized Markdown"]
    TWIN -->|Socket.IO| UI
    COORD -->|streamed answer| UI
```

- **`aegis_core/`** - the domain logic as an installable package: twin and routing, sensor simulation and fusion, anomaly and trend detection, emergency NLP, the event bus that ties modules together, the agents and LLM coordinator, the vision pipeline and the guided scenario.
- **`command_center/`** - the Flask + Socket.IO web app on top of it: auth, CSRF, rate limiting, the pages, and the tests.
- One process holds the live state in memory, so it runs as **one** gunicorn worker (threaded) - see [ROADMAP.md](ROADMAP.md#production-server-decision-2026-09-24).

---

## What's real and what's simulated

| Part | Status | Detail |
|---|---|---|
| Fire/smoke detection | **Real model** | Custom YOLOv8n trained on a public dataset (CPU); mAP50 0.576 validation / 0.421 held-out test ([metrics](#model-metrics)). Runs on a live webcam locally. |
| Object tracking, fall detection | **Real models** | YOLOv8n / YOLOv8n-pose on the live webcam (local only). |
| Aerial pose detection | **Real model, static images** | Custom YOLOv8n on SARD; recall 0.513 validation / 0.501 held-out test ([metrics](#model-metrics)). No drone - held-out test photos. |
| Report parsing | **Real** | spaCy NER + rule engine on whatever text you type. |
| Evacuation routing | **Real algorithm, modelled building** | Dijkstra over a 10-zone graph of a fictional building. |
| Sensor telemetry | **Simulated** | A simulator with realistic ramping; the fusion, anomaly and trend logic on top of it is real. |
| Chat answers | **Real LLM** (Groq / Claude) or **offline** | Agents read live system state; offline mode is keyword routing over the same agents. |
| Guided scenario | **Scripted inputs, real pipeline** | Sensor ramp, camera confidence and the caller report are scripted; everything downstream is the normal code path. Always labelled **SCENARIO / SIMULATED**. |
| Hosted demo | **No camera** | Cloud mode hides Live Vision; everything else works. |

---

## Model metrics

Every number below comes from a real Ultralytics validation run that anyone with the datasets can repeat:

```bash
command_center\venv\Scripts\python.exe command_center\tools\evaluate_models.py
```

It writes the raw results, including per-class numbers, to [`docs/metrics.json`](docs/metrics.json). The run used Ultralytics 8.4.144 on CPU at imgsz 416; a second run gave identical numbers. Where no labelled evaluation set exists, the table says **not measured** - nothing is estimated or copied from elsewhere.

| Model | Evaluation set | Images / objects | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| Fire/smoke (custom YOLOv8n) | validation split | 78 / 138 | 0.752 | 0.509 | 0.607 | 0.576 | 0.279 |
| Fire/smoke (custom YOLOv8n) | **held-out test split** | 44 / 83 | 0.564 | 0.409 | 0.474 | 0.421 | 0.177 |
| Aerial pose, SARD (custom YOLOv8n) | validation split | 401 / 1325 | 0.784 | 0.513 | 0.620 | 0.573 | 0.283 |
| Aerial pose, SARD (custom YOLOv8n) | **held-out test split** | 193 / 619 | 0.775 | 0.501 | 0.608 | 0.556 | 0.274 |
| Object tracking (stock YOLOv8n/s + ByteTrack) | - | - | not measured | not measured | not measured | not measured | not measured |
| Fall detection (YOLOv8n-pose + posture rule) | - | - | not measured | not measured | not measured | not measured | not measured |

- **How to read it.** Precision is the share of detections that were real. Recall is the share of real objects that were found. F1 = 2PR/(P+R). mAP50 summarises precision and recall over all confidence levels; mAP50-95 also demands tighter boxes. Precision and recall are reported at the confidence level that maximises F1 on each split, not at the app's alert thresholds.
- **Validation vs test.** Training picked the best checkpoint on the validation split, so validation numbers lean optimistic. The **test split was never used during training** - that's the honest number. The validation results reproduce the figures recorded when the models were trained: fire/smoke mAP50 0.576; aerial precision 0.784, recall 0.512, mAP50 0.572 in the final-epoch training log.
- **Why "not measured".** Tracking and fall detection use stock COCO-pretrained Ultralytics weights plus AegisAI's own logic: ByteTrack thresholds, and a posture rule on pose keypoints. This repo has no labelled video or fall/no-fall set to score them on. Ultralytics' published COCO benchmark measures the stock detector on COCO, not this pipeline, so it isn't quoted as if it were.
- **Weak classes.** In the aerial model, `Running` has 0 recall on both splits: it is never detected, and its precision of 1.0 only means it made no predictions at all. `laying_down` is the strongest class, with recall 0.85-0.88, which is the pose that matters most for finding injured people.

### What these numbers mean for AEGIS

- **Nothing a model sees changes the building on its own.** A camera detection only creates a *proposal* ("declare fire in Room201?"), and an operator must confirm it before the digital twin, routes or agents treat it as a fire. The numbers above are why that gate exists.
- **Fire/smoke: an early-warning input, not an alarm.** Test recall of 0.409 means the camera misses more than half of the labelled fire/smoke regions in unseen images, and a 44-image test set is too small to pin that down precisely. AEGIS therefore fuses the camera with sensor telemetry, where agreement between the two raises the risk score. The camera is never the only line of defence, and a real building's certified fire alarm remains the authority.
- **Aerial recall of 0.512 (0.501 on test) is acceptable only with human confirmation.** The model finds about half of the people in an aerial image. Its precision of about 0.78 means most of what it marks is a real person, so each detection is a useful lead for a human searcher. But a missed person is the most dangerous error in search and rescue, and roughly one in two is missed. So AEGIS presents aerial detections as leads for an operator to check, never as a head count and never as "area clear". A human reviews every image, and the absence of a detection proves nothing.
- **What would change this.** More and more varied training data, especially low-light fire, aerial `Running` and small distant people. Also an evaluation set recorded in the target environment, and labelled video to finally measure tracking and fall detection. Retraining is deliberately out of scope for now; see [ROADMAP.md](ROADMAP.md).

---

## Run it locally

Windows / PowerShell shown; macOS/Linux use `source venv/bin/activate`.

```powershell
git clone https://github.com/Aziz-ahmad555/AegisAI.git
cd AegisAI\command_center
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e ..
python app.py
```

Open http://127.0.0.1:5000 and sign in with **operator / aegisai2026** (local demo default). Then press **Run demo scenario** in the sidebar.

Live Vision needs the trained fire/smoke weights copied into `command_center/` as `fire_smoke_model.pt` (weights aren't committed - see [Model weights](#model-weights)); the other pages work without them.

### Free LLM for the command chat (Groq)

1. Create a free key at https://console.groq.com/keys.
2. Set it in the terminal you start the app from - in **single quotes**, and never paste it anywhere else:
   ```powershell
   $env:GROQ_API_KEY = 'gsk_...'
   ```
3. Start the app. The first line tells you which LLM is active, e.g. `LLM: groq / openai/gpt-oss-120b (auto-selected: available to this key)`; if the key is rejected it says so and the chat runs offline. The app picks a tool-capable Groq model your key can actually use.

Without a key the chat still works in **offline mode** (keyword routing over the same agents). Claude is also supported: set `ANTHROPIC_API_KEY` (default model `claude-sonnet-5`), or force a provider with `AEGISAI_LLM_PROVIDER=groq|claude|offline`.

### Your own password

Only a hash is stored. Generate it (you're prompted; nothing is echoed):
```powershell
python -c "import getpass; from werkzeug.security import generate_password_hash as h; print(h(getpass.getpass()))"
```
and set the output, in single quotes because it contains `$`: `$env:AEGISAI_PASSWORD_HASH = 'scrypt:...'`.

---

## Deploy it (free)

The repo ships a `Dockerfile` (cloud mode: no camera stack, gunicorn threaded worker, non-root user, health check) and a Render blueprint (`render.yaml`).

**Render (free web service):**
1. Push the repo to GitHub, then in Render: **New + → Blueprint** → pick the repo.
2. Render generates `AEGISAI_SECRET_KEY`. Fill in `AEGISAI_PASSWORD_HASH` (command above) and, optionally, `GROQ_API_KEY`.
3. Deploy. The free tier sleeps when idle; the first request after a sleep takes a while.

**Any Docker host:**
```bash
docker build -t aegisai .
docker run -p 8000:8000 -e AEGISAI_SECRET_KEY=<64 hex chars> -e AEGISAI_PASSWORD_HASH='<hash>' -e GROQ_API_KEY='<optional>' aegisai
```

Cloud mode **refuses to start** without a real `AEGISAI_SECRET_KEY` (32+ characters; `python -c "import secrets; print(secrets.token_hex(32))"`) and an `AEGISAI_PASSWORD_HASH`, and names what's missing.

---

## Configuration

All settings are environment variables; `command_center/.env.example` documents each one.

| Variable | Default | Purpose |
|---|---|---|
| `AEGISAI_PASSWORD_HASH` / `AEGISAI_USERNAME` | demo login | Operator credentials (hash required in cloud mode) |
| `AEGISAI_SECRET_KEY` | demo key | Session signing (required in cloud mode) |
| `AEGISAI_CLOUD_MODE` | `false` | Hosted mode: no camera, secure cookies, strict startup checks |
| `AEGISAI_TRUST_PROXY` | `false` | Use the real client IP behind a hosting proxy (rate limiting) |
| `GROQ_API_KEY` / `ANTHROPIC_API_KEY` | - | Chat LLM keys |
| `AEGISAI_LLM_PROVIDER` | auto | `groq`, `claude` or `offline` |
| `AEGISAI_GROQ_MODEL` / `AEGISAI_CLAUDE_MODEL` | auto / `claude-sonnet-5` | Pin a model |
| `AEGISAI_SENSOR_ZONE` / `AEGISAI_CAMERA_ZONE` | `CorridorB` / `Room201` | Zones the sensors and camera watch |
| `AEGISAI_SCENARIO_COUNTDOWN` | `10` | Seconds before the scenario auto-confirms |
| `AEGISAI_TRACK_CONF` / `AEGISAI_INFER_SIZE` / `AEGISAI_TRACK_MODEL` | `0.25` / `320` / `yolov8n.pt` | Live Vision tuning (chosen from a measured phone-detection test) |

---

## Security

- Hashed operator password, constant-time checks, session reset on login, login rate limiting (5 failures / 5 min per client).
- CSRF tokens on every POST; Socket.IO accepts same-origin connections only and refuses unauthenticated handshakes.
- A Content-Security-Policy with **no third-party origins** - every script, stylesheet and font is self-hosted.
- LLM output is untrusted: Markdown is rendered **server-side** with raw HTML disabled, then sanitized with an allowlist (no scripts, handlers, styles or images; safe links only). Tested against 19 XSS payloads and in a real browser.
- Caller text reaches the LLM only in fields marked `_untrusted`, and the prompt forbids following instructions inside them.
- Keys are never logged; anything key-shaped is redacted from error messages and tracebacks.

---

## Testing and CI

```powershell
pip install -r command_center/requirements-dev.txt
python -m pytest          # from the repo root - 255 tests
ruff check aegis_core command_center
```

GitHub Actions runs ruff and the full suite on every push, using the cloud requirement set (no camera or GPU stack). The tests use fake LLM clients - no network, no keys.

---

## Known limitations

- **Single building, single process.** One modelled building; live state lives in memory, so it runs as one worker and resets on restart. Multi-site or multi-worker would need Redis and persistent storage.
- **Simulated sensors.** Telemetry comes from a simulator, not real hardware.
- **Model accuracy.** Fire/smoke mAP50 is 0.576 on validation but 0.421 on the small held-out test set, and weaker in low light (daylight-skewed training data). Aerial recall is about 0.5 - it misses about half the people, so treat detections as leads, not counts. See [Model metrics](#model-metrics).
- **Fall detection** needs the full body in frame; a desk-height laptop webcam often can't see the hips.
- **Offline chat** is keyword routing - it answers from the right agents but can't reason across a nuanced question the way the LLM does.
- **Free hosting** sleeps when idle, and there's no camera there.
- **Not a certified safety system.** A portfolio project that demonstrates the architecture; it is not a replacement for a fire alarm system.

---

## Engineering findings

- **The live-vision lag was the server, not the model.** `eventlet` ran every thread as a green thread on one OS thread, so each ~200 ms YOLO inference froze the video stream, WebSocket and page loads, while buffered webcam frames piled up into 5-10 s of latency. Real threads plus a newest-frame-only grabber took it to ~120 ms.
- **"Building risk" must not be an average.** One fire among ten zones averaged to a green "normal"; the UI now headlines the worst zone.
- **Multimodal fusion reduces false confidence.** Camera-only detections are capped at moderate risk by design; only agreement between camera and sensors reaches CRITICAL.
- **Evacuation logic needs three outcomes, not two.** A room on fire can still leave through its own door (flagged), versus rooms that are genuinely isolated.
- **Bugs caught by testing in a real browser that unit tests couldn't:** a script-load-order bug that silently stopped evacuation routes from rendering, and an unauthenticated socket that stayed connected under a real server.
- **Model availability is per account.** A hard-coded Groq model returned 404 for a real key; the app now asks the provider which models the key can use.
- **Low-light / framing limits** of the detectors were measured, not assumed (per-keypoint confidence logging, a phone-in-hand capture to pick tracking thresholds).

### Model weights

Weights (`.pt`) aren't committed. They're reproducible from the dataset and training commands in [`archive/phases/`](archive/phases/README.md#running-an-individual-phase) (about 1-1.5 h each on a CPU laptop). The recorded metrics act as a lightweight model registry - see `archive/phases/phase11_multiagent_mlops_security/MLOPS.md`.

---

## Project structure

```
AegisAI/
  aegis_core/          domain logic (installable): twin + routing, fusion, NLP, event bus,
                       agents + LLM coordinator, vision pipeline, guided scenario
  command_center/      Flask + Socket.IO app: templates, static (design system, vendored libs),
                       tests, tools (detection diagnostic, model evaluation)
  archive/phases/      how it was built: 11 phase snapshots, per-phase instructions, training pipelines
  docs/                README screenshots, scenario GIF, metrics.json (model evaluation results)
  Dockerfile, render.yaml, ROADMAP.md
```

Built phase by phase - from a webcam detection script to a connected multimodal system; the full history is in [archive/phases](archive/phases/README.md) and the ongoing plan in [ROADMAP.md](ROADMAP.md).
