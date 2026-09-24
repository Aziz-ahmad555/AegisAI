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
- [ ] Serve static assets with cache headers; self-host / preconnect fonts; defer non-critical JS.
- [ ] Only broadcast state over WebSocket when it actually changed (twin) and throttle sensor broadcasts per page.
- [ ] Lazy-load heavy page components (3D hero on Overview) so first paint is instant.
- [x] Cloud mode never imports the vision stack (lazy import)
- [ ] Production server config: `gunicorn` with threaded worker (`gthread`), single worker (shared in-memory state).

## Stage 1 — Repository hygiene

- [x] Restore the main `README.md` to the repo root (it was moved into `phase1_vision/` by the restructure commit).
- [x] Datasets are no longer tracked. (They still live in old history, ~32 MB of `.git`; purging needs a history rewrite + force push — only if you want it.)
- [x] Re-encode `requirements.txt` as UTF-8 (it's UTF-16) and split: `requirements.txt` (full, local w/ vision) and `requirements-cloud.txt` (no torch/ultralytics).
- [x] Remove stray debug/output files (`phase10_digital_twin/output.txt`, `result.txt`, `debug_test.py`).
- [x] Add `.env.example` documenting every env var.
- [ ] Add `pyproject.toml` with ruff config + pre-commit hooks.

## Stage 2 — Make it one connected system (biggest credibility gain)

- [ ] **Shared core package** `aegis_core/` — single source of truth for fusion, anomaly, trend, NLP, routing, twin; remove the copy-pasted modules across phase folders and `command_center/`.
- [ ] **Event bus** — in-process pub/sub; every module publishes typed events (`FireDetected`, `SensorAnomaly`, `ReportParsed`, `ZoneCleared`…).
- [ ] **Live agents** — Fire/Medical/Route agents read the real digital twin + sensor state instead of hard-coded demo data.
- [ ] **Camera → fusion** — fire/smoke confidence from Live Vision feeds the fused risk score (currently always 0).
- [ ] **Sensors → twin** — sensors are mapped to zones; HIGH/CRITICAL fused risk raises that zone's risk on the twin.
- [ ] **Reports → twin** — a parsed report ("fire in CorridorA, 3 trapped") updates the twin and Medical agent automatically (with operator confirmation).
- [ ] **Incident timeline** — one unified, time-ordered log of every event with source, confidence, and resulting action.
- [ ] **Room-level people tracking** — trapped-people counts per zone feed the Medical agent and routing priority.

## Stage 3 — Tests & CI

- [ ] pytest suite: routing (clean / through-hazard / fully isolated), NLP extraction edge cases, fusion scoring & risk bands, anomaly detector, trend predictor, twin thread-safety.
- [ ] Flask tests: every route requires login, WebSocket actions reject unauthenticated sessions, API input validation.
- [ ] Coordinator tests: offline keyword routing; LLM path with a mocked client; tool-loop termination.
- [ ] GitHub Actions: ruff + pytest on every push; badge in README.

## Stage 4 — Security hardening

- [ ] Refuse to start in cloud mode with default password / secret key.
- [ ] Hash operator password (werkzeug `generate_password_hash`), constant-time compare.
- [ ] Login rate limiting / lockout.
- [ ] CSRF protection on the login form and JSON POST endpoints.
- [ ] Restrict Socket.IO CORS to the app's own origin.
- [ ] Input size limits on `/api/chat` and `/api/analyze_report`; validate zone/room names on socket events.
- [ ] Secure session cookies (`HttpOnly`, `SameSite`, `Secure` in cloud mode); security headers.
- [ ] Prompt-injection guard: the coordinator treats report text as data, never instructions.

## Stage 5 — LLM & agent upgrades

- [ ] Update to a current Claude model; model id configurable via env var.
- [ ] Stream chat responses token-by-token to the UI.
- [ ] Cap tool-use loop iterations; timeouts; show which agents were consulted in the UI.
- [ ] Conversation memory within a session ("and what about Room202?").
- [ ] Every agent answer cites the state it used (zone, reading, timestamp).

## Stage 6 — Frontend redesign & UX

- [ ] Shared design system: CSS tokens, type scale, spacing, component classes (cards, buttons, badges, stat tiles, tables) in one static stylesheet instead of per-page `<style>` blocks.
- [ ] Global live status bar (risk level, active incidents, connection state, LLM online/offline).
- [ ] Toast notifications for new critical events on every page.
- [ ] Real loading / empty / error / reconnecting states on every screen.
- [ ] Responsive layout (tablet + phone), collapsible sidebar.
- [ ] Light theme toggle in addition to the dark theme.
- [ ] Keyboard command palette (Ctrl+K) for navigation and quick actions.
- [ ] Micro-interactions: animated risk gauges, smooth graph transitions, route path animation.
- [ ] Accessibility: focus states, ARIA labels, contrast checks, reduced-motion support.
- [ ] Polished login screen.

## Stage 7 — Demo experience

- [ ] **Guided scenario mode** — one button plays a scripted incident end-to-end (sensors rise → camera confirms → zone ignites → reroute → agents brief the operator) with a narrated timeline.
- [ ] Scenario library: kitchen fire, blocked stairwell, medical emergency, multi-zone fire.
- [ ] Reset-to-baseline button.
- [ ] Incident report export (PDF/Markdown) summarizing a scenario.

## Stage 8 — Models (optional, longer)

- [ ] Retrain fire/smoke with low-light augmentation; before/after metrics table.
- [ ] Improve aerial recall (more epochs / larger imgsz); publish confusion matrices.
- [ ] Export to ONNX / OpenVINO for 2–3× faster CPU inference.
- [ ] Record precision/recall/F1 for every model in the README registry table.

## Stage 9 — Deploy & present

- [ ] Finish cloud mode (vision disabled, lightweight requirements) and deploy (Render / Railway / Fly.io).
- [ ] Dockerfile + one-command local run.
- [ ] README overhaul: live demo link, architecture diagram, screenshots/GIF, "real vs. simulated" table, known limitations, engineering findings.
- [ ] Short demo video.

---

### Execution order
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 → 8

Each stage ships as small, working commits; nothing is labeled done until it's been run and verified.
