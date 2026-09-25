# AegisAI — Polish & Upgrade Roadmap

Goal: turn AegisAI from a set of well-built phase demos into **one connected, fast, tested, deployed system** that reads as a considered product.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Stage 0 — Performance & responsiveness (do first: it's what users feel)

**Root causes found for the 5–10 s Live Vision lag and general sluggishness:**
- `eventlet.monkey_patch()` turns every thread into a cooperative green thread on a single OS thread. Each YOLO inference (100–300 ms of CPU-bound C code) blocks *everything* — the MJPEG stream, WebSocket broadcasts, and every HTTP request — until it finishes.
- The webcam driver buffers frames. The loop reads one frame, spends ~200 ms on inference, then reads the *next buffered* (already old) frame. Latency accumulates to seconds.
- Capture, inference, and streaming all run in one loop with a fixed sleep, so the slowest step dictates the latency of everything.

Tasks:
- [x] Drop eventlet; use Flask-SocketIO `threading` mode (real OS threads; inference releases the GIL in PyTorch/OpenCV).
- [x] Split vision into a **grabber thread** (reads the camera continuously, keeps only the newest frame, buffer size 1) and an **inference thread** (always processes the newest frame, drops stale ones).
- [x] The MJPEG generator waits on a frame-ready condition instead of a fixed sleep, so new frames go out immediately.
- [x] Cap PyTorch threads so inference doesn't starve the web server; warm up models at startup so the first frame isn't slow.
- [x] Show live FPS + end-to-end latency in the vision status line (measured, not guessed).
- [x] Static assets cached (1 h, URL-versioned by file mtime so edits show up at once); fonts and all JS libraries self-hosted; the 3D map's library lazy-loads.
- [x] Twin state is only re-broadcast when it changes (sensor readings still stream every 2 s by design).
- [x] The Overview 3D map loads three.js only when scrolled into view.
- [x] Cloud mode never imports the vision stack (lazy import)
- [x] Production server config — decided and verified (see **Production server** below).
- [x] Live Vision tracking thresholds configurable (`AEGISAI_TRACK_CONF`, `AEGISAI_INFER_SIZE`, `AEGISAI_TRACK_MODEL`); overlay no longer collides with box labels; `tools/diagnose_detections.py` for raw per-class confidence + latency.
- [x] Tracking defaults chosen from a real phone-in-hand capture (151 lit frames, dark screen, fingers over it): **keep yolov8n @ 320, threshold 0.25.** yolov8n@320 finds the phone in 142/151 frames (median 0.72) and the full app tracking path keeps it in 139/151; yolov8s@320 matches that hit rate with higher confidence (0.82) but ~2x inference cost and recurring false 'bird' boxes; 416/640 add latency and false positives (yolov8n@640: 'bird' in every frame). Lowering to 0.20 gains 2 frames, raising to 0.30 loses 1. The earlier live miss could not be reproduced on these frames.
- [x] Diagnostic warm-up waits for *lit* frames (this webcam delivers ~5 s of black frames after opening) and excludes black frames from the stats.

### Production server (decision, 2026-09-24)

**Decision:** Flask-SocketIO in `threading` async mode, served by **gunicorn's threaded worker with exactly one worker process**:

```
pip install -r command_center/requirements-cloud.txt && pip install .
gunicorn -k gthread -w 1 --threads 50 --chdir command_center -b 0.0.0.0:$PORT app:app
```

**Why:**
- eventlet was removed because its green threads serialize CPU-bound YOLO inference with every request (the root cause of the lag). Threading mode uses real OS threads.
- WebSockets still work in production: `simple-websocket` has a gunicorn mode that takes over the raw socket from `environ['gunicorn.socket']`, so there's no silent fallback to long-polling.
- **One worker is required, not a tuning choice:** the twin, sensor state, event bus and pending actions live in process memory, and gunicorn has no sticky sessions for Socket.IO. Scale with `--threads`, not `-w`. Going multi-worker would need Redis (`message_queue=`) plus moving state out of process — out of scope for now.
- Every open page holds one thread (WebSocket), plus one per Live Vision MJPEG viewer; 50 threads covers a demo comfortably.
- Local dev keeps `python app.py` (Werkzeug + `allow_unsafe_werkzeug=True`, which is fine on 127.0.0.1). gunicorn does not run natively on Windows.

**Verified** in a `python:3.11-slim` container with `requirements-cloud.txt` and `AEGISAI_CLOUD_MODE=true`: all pages 3–12 ms, Socket.IO upgraded to a real `websocket` transport, live pushes delivered, torch/ultralytics not installed.
That check also exposed that **unauthenticated sockets stayed connected** under a real server (calling `disconnect()` inside the connect handler doesn't reject the handshake) — fixed by raising `ConnectionRefusedError`.

## Stage 1 — Repository hygiene

- [x] Restore the main `README.md` to the repo root (it was moved into `phase1_vision/` by the restructure commit).
- [x] Datasets are no longer tracked. (They still live in old history, ~32 MB of `.git`; purging needs a history rewrite + force push — only if you want it.)
- [x] Re-encode `requirements.txt` as UTF-8 (it's UTF-16) and split: `requirements.txt` (full, local w/ vision) and `requirements-cloud.txt` (no torch/ultralytics).
- [x] Remove stray debug/output files (`phase10_digital_twin/output.txt`, `result.txt`, `debug_test.py`).
- [x] Add `.env.example` documenting every env var.
- [x] `pyproject.toml` (package + pytest + ruff config) and a ruff pre-commit hook.

## Stage 2 — Make it one connected system (biggest credibility gain)

- [x] **Shared core package** `aegis_core/` — single source of truth for fusion, anomaly, trend, NLP, routing, twin, agents, events and vision; installable via the root `pyproject.toml`. Phase folders were moved (not deleted) to `archive/phases/` as historical snapshots.
- [x] **Event bus** — `events.py`: in-process pub/sub with typed events; failing subscribers are isolated.
- [x] **Live agents** — Fire/Medical/Route agents read the live twin, sensor, camera and report state; unknowns are reported as unknown (e.g. medical unit availability), report data is labelled unverified.
- [x] **Camera → fusion** — fire/smoke confidence from Live Vision (fire/smoke mode) feeds the fused risk score; detections with hysteresis raise a timeline event and a *proposed* fire declaration for `AEGISAI_CAMERA_ZONE`.
- [x] **Sensors → twin** — fused sensor risk is mirrored onto `AEGISAI_SENSOR_ZONE` (amber on the map, doesn't block routes); level changes and anomalies go to the timeline; CRITICAL proposes a fire declaration.
- [x] **Reports → twin** — reports are matched to twin zones; fire reports create a proposed action the operator confirms or dismisses (tray on every page); people counts feed the Medical agent.
- [x] **Incident timeline** — unified, time-ordered, pushed live over the WebSocket; on the Digital Twin page, with toasts for warnings/critical events on every page.
- [ ] **Room-level people tracking** — trapped-people counts per zone feed the Medical agent and routing priority.

## Stage 3 — Tests & CI

- [x] pytest suite: routing (clean / through-hazard / fully isolated), NLP extraction edge cases, fusion scoring & risk bands, anomaly detector, trend predictor, twin thread-safety. Found and fixed two NLP bugs ('stuck' reports rated LOW; 'water is rising' not a flood).
- [x] Flask tests: every route requires login, WebSocket actions reject unauthenticated sessions, API input validation.
- [x] Coordinator tests: offline keyword routing; LLM path with a fake client; tool-loop termination (loop now capped at 5 rounds).
- [x] GitHub Actions: ruff + pytest on every push (cloud requirement set, Python 3.11); badge in README.

## Stage 4 — Security hardening

- [x] Refuse to start in cloud mode without a real secret key (32+ chars) and a password hash; local mode keeps the zero-setup demo login.
- [x] Hashed operator password (`AEGISAI_PASSWORD_HASH`, one-line generator in README/.env.example), constant-time compare, session cleared on login/logout.
- [x] Login rate limiting: 5 failures / 5 min per client address -> 429 + Retry-After; `AEGISAI_TRUST_PROXY` for real client IPs behind a proxy.
- [x] CSRF protection on the login form and all POST endpoints (session token; header added to every fetch()).
- [x] Socket.IO restricted to the app's own origin (`AEGISAI_ALLOWED_ORIGINS` to extend); verified a foreign origin is rejected.
- [x] Input size limits (64 KB body, 2000-char reports, 1000-char questions); zone/room names validated on socket events.
- [x] Session cookies HttpOnly + SameSite=Lax (+ Secure in cloud mode); nosniff, X-Frame-Options, Referrer-Policy, Content-Security-Policy.
- [x] Prompt-injection guard: caller text reaches the LLM only in fields marked `_untrusted`, and the coordinator prompt forbids acting on it (tested). A red-team pass belongs with the Stage 5 LLM work.

## Stage 5 — LLM & agent upgrades

- [x] Current Claude model (default `claude-sonnet-5`, verified against Anthropic's model docs); `AEGISAI_CLAUDE_MODEL` / `AEGISAI_CLAUDE_EFFORT` configurable; server-side refusal fallbacks on by default for Opus 5 / Fable 5.1, opt-in otherwise.
- [x] Chat answers stream over Socket.IO to the asking browser only; refusals and mid-stream failures discard partial text and fall back to the offline summary.
- [x] 5-round tool cap kept; truncated tool calls never run; tool input validated; 60 s client timeout; consulted agents shown as chips on each answer.
- [x] Conversation memory: last 6 Q/A pairs per login (server-side, bounded); "New conversation" clears it.
- [x] Groq provider (`AEGISAI_LLM_PROVIDER` = groq | claude | offline; groq is the default when `GROQ_API_KEY` is set) with the same streaming, 5-round cap, memory, fallbacks and injection guard; startup `LLM:` line and chat badge show the active provider.
- [x] Evidence citations: every answer lists what its consulted agents reported (source sensor/camera/report/twin, zone, value, time), built from the same data the model received - never model-written; caller text never repeated; works with Groq, Claude and offline. Verified with a real Groq answer.

## Stage 6 — Frontend redesign & UX

- [x] Chat renders model Markdown safely: server-side markdown-it (raw HTML off) + nh3 allowlist, no images, safe links; streamed as sanitized HTML; XSS payloads tested in pytest and in a real browser (nothing executed).

- [x] Shared design system (`static/css/aegis.css`): tokens, 4px spacing scale, type scale (12px min), components; no per-page `<style>` blocks remain. Red reserved for danger states (tested).
- [x] Global status bar on every page: threat level first, active fires, alerts awaiting confirmation, sensors, link + chat LLM.
- [x] Toast notifications for warnings/critical events on every page.
- [x] Loading / empty / error / link-down states on the live screens (incidents, timeline, report analyzer, chat, vision).
- [x] Responsive layout (sidebar becomes a scrolling top nav below 900px; verified no horizontal overflow at 375px).
- [ ] Light theme toggle in addition to the dark theme. *(not started - dark-only by design choice for now)*
- [ ] Keyboard command palette (Ctrl+K) for navigation and quick actions. *(not started)*
- [ ] Micro-interactions: animated risk gauges, smooth graph transitions, route path animation. *(not started)*
- [x] Accessibility basics: visible focus, labelled controls/regions, 4.5:1 text contrast, reduced-motion support. (No full audit yet.)
- [x] Polished login screen.

## Stage 7 — Demo experience

- [x] **Guided scenario mode** — 'Run demo scenario' plays a scripted incident through the real modules (sensor ramp → simulated camera → caller report → proposal with visible countdown/auto-confirm → twin + rerouting → agents' briefing); always labelled SCENARIO / SIMULATED; Pause/Resume, Stop, Reset; works in offline LLM mode.
- [x] Scenario library: corridor fire, kitchen fire, smoke-logged stairwell, medical emergency. Chosen from the sidebar; routes are compared with the pre-incident routes, so reroutes are reported. *(Multi-zone fire not yet.)*
- [x] Reset-to-baseline (fires, proposals, reports, sensors and camera back to normal).
- [x] Incident report export (Markdown): summary, decisions, caller reports, current routes and the timeline since the last reset, built by code with every external string escaped; labelled SIMULATED for scenario data. *(PDF not yet.)*

## Stage 8 — Models (optional, longer)

- [ ] Retrain fire/smoke with low-light augmentation; before/after metrics table.
- [ ] Improve aerial recall (more epochs / larger imgsz); publish confusion matrices.
- [ ] Export to ONNX / OpenVINO for 2–3× faster CPU inference.
- [x] Precision/recall/F1/mAP for every model in the README, from a reproducible evaluation run (`command_center/tools/evaluate_models.py` -> `docs/metrics.json`, guarded by a test); held-out test split added; tracking and fall detection honestly marked "not measured" (no labelled data).

## Stage 9 — Deploy & present

- [x] Cloud mode finished; deploy artifacts ready (`Dockerfile`, `render.yaml`, `/healthz`), verified in a container with a real browser. Live at https://aegisai-9aki.onrender.com (Render free tier).
- [x] Dockerfile (cloud mode, non-root, health check, gunicorn gthread x1).
- [x] README overhaul: architecture diagram, screenshots + scenario GIF, real-vs-simulated table, known limitations, engineering findings, local + cloud setup with free Groq.
- [ ] Short demo video.

---

### Execution order
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 → 8

Each stage ships as small, working commits; nothing is labeled done until it's been run and verified.
