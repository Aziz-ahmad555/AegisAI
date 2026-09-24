import os
import secrets
import threading
import time
from collections import OrderedDict
from functools import wraps

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from flask_socketio import SocketIO, disconnect, emit
from werkzeug.middleware.proxy_fix import ProxyFix

from aegis_core import coordinator
from aegis_core.building_state import BuildingDigitalTwin
from aegis_core.coordinator import init_agents
from aegis_core.emergency_nlp import parse_emergency_report
from aegis_core.sensor_state import SensorFusionState
from aegis_core.system import AegisSystem

import safe_markdown
import security

# When true (set via env var on the hosted deployment), Live Vision is hidden
# entirely rather than attempting to stream from a camera that doesn't exist
# on a cloud server. Also skips loading the vision models at all, saving
# meaningful memory on constrained hosting tiers.
CLOUD_MODE = os.environ.get("AEGISAI_CLOUD_MODE", "false").lower() == "true"

# Local mode runs with simple documented defaults; a hosted deployment must be
# configured properly or it refuses to start (see security.py).
if CLOUD_MODE:
    _problems = security.cloud_config_problems(os.environ)
    if _problems:
        raise security.InsecureCloudConfig(
            "Refusing to start in cloud mode:\n  - " + "\n  - ".join(_problems)
        )

credentials = security.Credentials.from_env(os.environ)
login_limiter = security.LoginRateLimiter()

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("AEGISAI_SECRET_KEY") or security.DEFAULT_SECRET_KEY,
    MAX_CONTENT_LENGTH=security.MAX_REQUEST_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=CLOUD_MODE,      # hosted = HTTPS; local http://127.0.0.1 must still work
    # Static assets (vendored libraries, fonts, CSS, JS) are cacheable; Flask
    # still revalidates with ETag/Last-Modified after this, so edits show up.
    SEND_FILE_MAX_AGE_DEFAULT=3600,
)
if os.environ.get("AEGISAI_TRUST_PROXY", "false").lower() == "true":
    # Behind a hosting proxy every request comes from the proxy's IP; trust its
    # X-Forwarded-* headers so rate limiting keys on the real client address.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Socket.IO accepts only same-origin browsers unless extra origins are listed
# (comma-separated), which blocks cross-site WebSocket hijacking.
_allowed_origins = [o.strip() for o in os.environ.get("AEGISAI_ALLOWED_ORIGINS", "").split(",") if o.strip()]

# Threading mode (real OS threads) rather than eventlet: eventlet's green
# threads all share one OS thread, so every CPU-bound YOLO inference froze the
# whole server - video stream, WebSocket pushes and page loads - until it
# finished. PyTorch and OpenCV release the GIL during inference, so plain
# threads let vision and the web server genuinely run in parallel.
# Production: see "Production server" in ROADMAP.md (gunicorn gthread, 1 worker).
socketio = SocketIO(app, cors_allowed_origins=_allowed_origins or None, async_mode="threading")

twin = BuildingDigitalTwin()
# Chat LLM: groq / claude / offline, chosen from the environment (see
# coordinator.get_llm). One status line at startup - never the key itself.
llm_client, LLM_DESCRIPTION = coordinator.get_llm()
print(f"LLM: {LLM_DESCRIPTION}", flush=True)
sensors = SensorFusionState()
if CLOUD_MODE:
    vision = None
else:
    # Imported lazily so hosted deployments never pull in torch/ultralytics.
    from aegis_core.vision_stream import VisionStream
    vision = VisionStream()

# One object wiring the modules together (camera -> fusion -> twin, reports
# -> proposed actions, everything -> incident timeline); agents read it live.
system = AegisSystem(twin, sensors, vision)
init_agents(system)


@app.context_processor
def inject_template_globals():
    return {
        "cloud_mode": CLOUD_MODE,
        "csrf_token": security.csrf_token,
        # Only advertise the built-in demo login when it's actually in use.
        "show_default_login": not CLOUD_MODE and credentials.uses_default_password,
        "llm_label": llm_client.describe() if llm_client else "offline",
    }


app.before_request(security.verify_csrf)
app.after_request(security.add_security_headers)


@app.url_defaults
def static_cache_bust(endpoint, values):
    # Static files are cached for an hour; versioning each URL by the file's
    # modification time means an edited CSS/JS file gets a new URL at once,
    # while unchanged files keep being served from the browser cache.
    if endpoint == "static" and "filename" in values:
        path = os.path.join(app.static_folder, values["filename"])
        try:
            values["v"] = int(os.stat(path).st_mtime)
        except OSError:
            pass


@app.errorhandler(400)
@app.errorhandler(413)
@app.errorhandler(429)
def api_error(err):
    # JSON for API calls, plain text otherwise; never a stack trace.
    if request.path.startswith("/api/"):
        return jsonify({"error": err.description}), err.code
    return err.description, err.code


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    status = 200
    if request.method == "POST":
        client = request.remote_addr or "unknown"
        wait = login_limiter.retry_after(client)
        if wait:
            error = f"Too many failed attempts. Try again in {wait} seconds."
            response = render_template("login.html", error=error)
            return response, 429, {"Retry-After": str(wait)}
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if credentials.verify(username, password):
            login_limiter.reset(client)
            # New session on login (prevents session fixation); a fresh CSRF
            # token is issued on the next page render.
            session.clear()
            session["logged_in"] = True
            # Created here (HTTP), not in a socket handler: Flask-SocketIO keeps
            # a per-connection copy of the session, so an id minted there would
            # never reach the browser's cookie and the pages couldn't share it.
            session["chat_id"] = secrets.token_urlsafe(16)
            return redirect(url_for("overview"))
        login_limiter.record_failure(client)
        error = "Invalid credentials"
        status = 401
    return render_template("login.html", error=error), status


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def overview():
    return render_template("overview.html")


@app.route("/twin")
@login_required
def twin_page():
    return render_template("twin.html")


@app.route("/nlp")
@login_required
def nlp_page():
    return render_template("nlp.html")


@app.route("/chat")
@login_required
def chat_page():
    _chat_id()   # sessions from before chat memory existed get their id here
    return render_template("chat.html", llm=llm_client, llm_description=LLM_DESCRIPTION)


@app.route("/sensors")
@login_required
def sensors_page():
    return render_template("sensors.html", sensor_zone=system.sensor_zone())


@app.route("/aerial")
@login_required
def aerial_page():
    return render_template("aerial.html", samples=AERIAL_SAMPLES)


@app.route("/vision")
@login_required
def vision_page():
    if CLOUD_MODE:
        return render_template("vision_disabled.html")
    vision.start()
    return render_template("vision.html", camera_zone=system.camera_zone())


@app.route("/video_feed")
@login_required
def video_feed():
    if CLOUD_MODE:
        return "", 404
    return Response(vision.generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/vision_mode", methods=["POST"])
@login_required
def api_vision_mode():
    if CLOUD_MODE:
        return jsonify({"success": False, "reason": "disabled in hosted demo"}), 404
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "tracking")
    ok = vision.set_mode(mode)
    return jsonify({"success": ok, "mode": mode})


@app.route("/api/vision_info")
@login_required
def api_vision_info():
    if CLOUD_MODE:
        return jsonify({"detail": "Live Vision is disabled in the hosted demo - run locally to use this feature."})
    return jsonify(vision.get_info())


@app.route("/api/analyze_report", methods=["POST"])
@login_required
def api_analyze_report():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "No text provided"}), 400
    if len(text) > security.MAX_REPORT_CHARS:
        return jsonify({"error": f"Report too long (max {security.MAX_REPORT_CHARS} characters)"}), 413
    result = parse_emergency_report(text)
    linked = system.ingest_report(result)
    result["zones"] = linked["zones"]
    result["proposals"] = linked["proposals"]
    return jsonify(result)


@app.route("/api/timeline")
@login_required
def api_timeline():
    return jsonify({"events": system.bus.timeline(limit=100), "pending_actions": system.pending_actions()})


@app.route("/api/actions/<int:proposal_id>", methods=["POST"])
@login_required
def api_decide_action(proposal_id):
    data = request.get_json(force=True, silent=True) or {}
    decided = system.decide(proposal_id, bool(data.get("approve")))
    if decided is None:
        return jsonify({"error": "No such pending action (already decided?)"}), 404
    broadcast_twin_state()
    return jsonify({"decided": decided, "approved": bool(data.get("approve"))})


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    # Blocking form (scripts, tests); the chat page streams over Socket.IO.
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    error = _question_error(question)
    if error:
        return jsonify({"error": error[0]}), error[1]
    answer = coordinator.ask_coordinator(question, llm_client, _chat_history())
    _remember_turn(question, answer)
    mode = "llm" if llm_client else "offline"
    provider = llm_client.describe() if llm_client else LLM_DESCRIPTION
    return jsonify({"answer": answer, "mode": mode, "provider": provider})


# --- chat memory ---------------------------------------------------------------
# Short per-login conversation memory ("and what about Room202?"). Kept
# server-side and bounded - only question/answer text, never tool data.

_chat_histories = OrderedDict()      # chat_id -> list of {"role", "content"}
_chat_lock = threading.Lock()
MAX_CHAT_SESSIONS = 200
CHAT_RENDER_INTERVAL = 0.08   # seconds between streamed re-renders (~12/s)


def _question_error(question):
    if not isinstance(question, str) or not question.strip():
        return "No question provided", 400
    if len(question) > security.MAX_QUESTION_CHARS:
        return f"Question too long (max {security.MAX_QUESTION_CHARS} characters)", 413
    return None


def _chat_id():
    if "chat_id" not in session:
        session["chat_id"] = secrets.token_urlsafe(16)
    return session["chat_id"]


def _chat_history(chat_id=None):
    with _chat_lock:
        return list(_chat_histories.get(chat_id or _chat_id(), []))


def _remember_turn(question, answer, chat_id=None):
    chat_id = chat_id or _chat_id()
    if not answer.strip():
        return
    with _chat_lock:
        turns = _chat_histories.pop(chat_id, [])
        turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        _chat_histories[chat_id] = turns[-2 * coordinator.HISTORY_TURNS:]
        while len(_chat_histories) > MAX_CHAT_SESSIONS:
            _chat_histories.popitem(last=False)


@app.route("/api/chat/reset", methods=["POST"])
@login_required
def api_chat_reset():
    with _chat_lock:
        _chat_histories.pop(_chat_id(), None)
    return jsonify({"ok": True})


AERIAL_SAMPLES = [
    {"file": "sample_1.jpg", "caption": "Aerial survey - person detected in open terrain"},
    {"file": "sample_2.jpg", "caption": "Aerial survey - pose classification in progress"},
    {"file": "sample_3.jpg", "caption": "Aerial survey - distress-relevant pose detection"},
    {"file": "sample_4.jpg", "caption": "Aerial survey - search and rescue scenario"},
    {"file": "sample_5.jpg", "caption": "Aerial survey - wooded terrain detection"},
]


# --- WebSocket handlers ---

def broadcast_twin_state():
    socketio.emit("state_update", twin.get_state_snapshot())


def broadcast_sensor_state():
    socketio.emit("sensor_update", sensors.get_snapshot())


def _push_event(event):
    # Every timeline event goes to every connected operator immediately, plus
    # the current pending-action list so confirm/dismiss controls stay in sync.
    socketio.emit("timeline_event", event.to_dict())
    socketio.emit("pending_actions", system.pending_actions())


system.bus.subscribe(_push_event)


@socketio.on("connect")
def handle_connect(auth=None):
    if not session.get("logged_in"):
        # Refuse the handshake itself. Calling disconnect() here doesn't
        # reliably drop a connection that is still being established under a
        # real server (verified under gunicorn): the client stayed connected
        # and received every broadcast.
        raise ConnectionRefusedError("unauthorized")
    # Initial state goes to the connecting client only, not everyone.
    emit("state_update", twin.get_state_snapshot())
    emit("sensor_update", sensors.get_snapshot())
    emit("timeline", system.bus.timeline(limit=50))
    emit("pending_actions", system.pending_actions())


@socketio.on("trigger_fire")
def handle_trigger_fire(data):
    if not session.get("logged_in"):
        disconnect()
        return
    zone = (data or {}).get("zone")
    if zone and system.start_fire(zone, source="operator"):
        broadcast_twin_state()


@socketio.on("clear_zone")
def handle_clear_zone(data):
    if not session.get("logged_in"):
        disconnect()
        return
    zone = (data or {}).get("zone")
    if zone and system.clear_zone(zone, source="operator"):
        broadcast_twin_state()


@socketio.on("request_route")
def handle_request_route(data):
    if not session.get("logged_in"):
        disconnect()
        return
    start = (data or {}).get("start", "Room101")
    if start not in system.zones():
        return
    # Reply to the operator who asked, not to every connected client.
    emit("route_result", {"start": start, "route": twin.compute_evacuation_route(start)})


@socketio.on("decide_action")
def handle_decide_action(data):
    if not session.get("logged_in"):
        disconnect()
        return
    try:
        proposal_id = int((data or {}).get("id"))
    except (TypeError, ValueError):
        return
    if system.decide(proposal_id, bool(data.get("approve"))) is not None:
        broadcast_twin_state()


@socketio.on("chat_ask")
def handle_chat_ask(data):
    if not session.get("logged_in"):
        disconnect()
        return
    question = (data or {}).get("question", "")
    sid = request.sid
    error = _question_error(question)
    if error:
        emit("chat_event", {"type": "error", "text": error[0]})
        return
    chat_id = _chat_id()
    history = _chat_history(chat_id)

    def run():
        # Streams to the asking browser only; the answer (minus anything a
        # refusal/failure told us to discard) becomes conversation memory.
        # Model text is untrusted: the browser only ever receives it as
        # server-sanitized HTML (safe_markdown), re-rendered as it grows so
        # Markdown displays correctly mid-stream.
        chunks = []
        last_render = 0.0

        def push_html():
            html = safe_markdown.render("".join(chunks).lstrip("\n"))
            socketio.emit("chat_event", {"type": "html", "html": html}, to=sid)

        try:
            for event in coordinator.iter_coordinator(question, llm_client, history):
                if event["type"] == "text":
                    chunks.append(event["text"])
                    if time.monotonic() - last_render >= CHAT_RENDER_INTERVAL:
                        push_html()
                        last_render = time.monotonic()
                    continue
                if event["type"] == "reset":
                    chunks = []
                    push_html()
                    continue
                socketio.emit("chat_event", event, to=sid)
            push_html()                   # final, complete render
            _remember_turn(question, "".join(chunks).lstrip("\n"), chat_id)
        except Exception as e:           # never leave the operator's UI hanging
            print(f"[chat] failed: {e}")
            socketio.emit("chat_event", {"type": "error", "text": "The coordinator failed; please retry."}, to=sid)
        finally:
            socketio.emit("chat_event", {"type": "done"}, to=sid)

    socketio.start_background_task(run)


@socketio.on("trigger_sensor_event")
def handle_trigger_sensor_event():
    if not session.get("logged_in"):
        disconnect()
        return
    sensors.trigger_event()


def periodic_broadcast():
    # Sensors change every tick (live readings), so they always go out. The
    # twin only changes on events - which already broadcast immediately - or
    # when monitored risk moves, so it's only re-sent when its state differs.
    last_twin = None
    while True:
        time.sleep(2)
        snapshot = twin.get_state_snapshot()
        if snapshot != last_twin:
            socketio.emit("state_update", snapshot)
            last_twin = snapshot
        broadcast_sensor_state()


# Start background loops at import time, not just under __main__, so they
# also run correctly under a production WSGI server (gunicorn) which imports
# this module directly rather than executing it as a script.
sensors.start_background_loop(
    interval_seconds=1.0,
    camera_provider=system.camera_confidence,
    on_update=system.on_sensor_update,
)
_broadcaster_thread = threading.Thread(target=periodic_broadcast, daemon=True)
_broadcaster_thread.start()


if __name__ == "__main__":
    print("AegisAI Command Center")
    if credentials.uses_default_password:
        print(f"Login: {credentials.username} / {security.DEFAULT_PASSWORD}  (local demo default)")
    else:
        print(f"Login: username '{credentials.username}' with your configured password")
    print("Open http://127.0.0.1:5000 in your browser")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
