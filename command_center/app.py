import os
import threading
import time
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_socketio import SocketIO, disconnect, emit
from aegis_core.building_state import BuildingDigitalTwin
from aegis_core.emergency_nlp import parse_emergency_report
from aegis_core.coordinator import ask_coordinator, get_client, init_agents
from aegis_core.sensor_state import SensorFusionState
from aegis_core.system import AegisSystem

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("AEGISAI_SECRET_KEY", "aegisai-command-center-demo-key")
# Threading mode (real OS threads) rather than eventlet: eventlet's green
# threads all share one OS thread, so every CPU-bound YOLO inference froze the
# whole server - video stream, WebSocket pushes and page loads - until it
# finished. PyTorch and OpenCV release the GIL during inference, so plain
# threads let vision and the web server genuinely run in parallel.
# Production: see "Production server" in ROADMAP.md (gunicorn gthread, 1 worker).
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# When true (set via env var on the hosted deployment), Live Vision is hidden
# entirely rather than attempting to stream from a camera that doesn't exist
# on a cloud server. Also skips loading the vision models at all, saving
# meaningful memory on constrained hosting tiers.
CLOUD_MODE = os.environ.get("AEGISAI_CLOUD_MODE", "false").lower() == "true"

OPERATOR_USERNAME = os.environ.get("AEGISAI_USERNAME", "operator")
OPERATOR_PASSWORD = os.environ.get("AEGISAI_PASSWORD", "aegisai2026")

twin = BuildingDigitalTwin()
llm_client = get_client()
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
def inject_cloud_mode():
    return {"cloud_mode": CLOUD_MODE}


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
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == OPERATOR_USERNAME and password == OPERATOR_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("overview"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
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
    llm_status = "connected" if llm_client else "offline (keyword-routing fallback)"
    return render_template("chat.html", llm_status=llm_status)


@app.route("/sensors")
@login_required
def sensors_page():
    return render_template("sensors.html")


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
    return render_template("vision.html")


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
    data = request.get_json(force=True)
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
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
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
    data = request.get_json(force=True) or {}
    decided = system.decide(proposal_id, bool(data.get("approve")))
    if decided is None:
        return jsonify({"error": "No such pending action (already decided?)"}), 404
    broadcast_twin_state()
    return jsonify({"decided": decided, "approved": bool(data.get("approve"))})


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(force=True)
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "No question provided"}), 400
    answer = ask_coordinator(question, llm_client)
    mode = "llm" if llm_client else "offline"
    return jsonify({"answer": answer, "mode": mode})


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


@socketio.on("trigger_sensor_event")
def handle_trigger_sensor_event():
    if not session.get("logged_in"):
        disconnect()
        return
    sensors.trigger_event()


def periodic_broadcast():
    while True:
        time.sleep(2)
        broadcast_twin_state()
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
    print(f"Login with username '{OPERATOR_USERNAME}' (set AEGISAI_USERNAME/AEGISAI_PASSWORD env vars to change)")
    print("Open http://127.0.0.1:5000 in your browser")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
