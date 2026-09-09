import os
import threading
import time
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, disconnect
from building_state import BuildingDigitalTwin
from emergency_nlp import parse_emergency_report
from coordinator import ask_coordinator, get_client
from sensor_state import SensorFusionState

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("AEGISAI_SECRET_KEY", "aegisai-command-center-demo-key")
socketio = SocketIO(app, cors_allowed_origins="*")

OPERATOR_USERNAME = os.environ.get("AEGISAI_USERNAME", "operator")
OPERATOR_PASSWORD = os.environ.get("AEGISAI_PASSWORD", "aegisai2026")

twin = BuildingDigitalTwin()
llm_client = get_client()
sensors = SensorFusionState()


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


@app.route("/api/analyze_report", methods=["POST"])
@login_required
def api_analyze_report():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    result = parse_emergency_report(text)
    return jsonify(result)


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


# --- Digital Twin WebSocket handlers ---

def broadcast_twin_state():
    snapshot = twin.get_state_snapshot()
    socketio.emit("state_update", snapshot)


def broadcast_sensor_state():
    snapshot = sensors.get_snapshot()
    socketio.emit("sensor_update", snapshot)


@socketio.on("connect")
def handle_connect():
    if not session.get("logged_in"):
        disconnect()
        return
    broadcast_twin_state()
    broadcast_sensor_state()


@socketio.on("trigger_fire")
def handle_trigger_fire(data):
    if not session.get("logged_in"):
        disconnect()
        return
    zone = data.get("zone")
    if zone:
        twin.start_fire(zone)
        broadcast_twin_state()


@socketio.on("clear_zone")
def handle_clear_zone(data):
    if not session.get("logged_in"):
        disconnect()
        return
    zone = data.get("zone")
    if zone:
        twin.clear_zone(zone)
        broadcast_twin_state()


@socketio.on("request_route")
def handle_request_route(data):
    if not session.get("logged_in"):
        disconnect()
        return
    start = data.get("start", "Room101")
    route = twin.compute_evacuation_route(start)
    socketio.emit("route_result", {"start": start, "route": route})


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


if __name__ == "__main__":
    sensors.start_background_loop(interval_seconds=1.0)

    broadcaster = threading.Thread(target=periodic_broadcast, daemon=True)
    broadcaster.start()

    print("AegisAI Command Center")
    print(f"Login with username '{OPERATOR_USERNAME}' (set AEGISAI_USERNAME/AEGISAI_PASSWORD env vars to change)")
    print("Open http://127.0.0.1:5000 in your browser")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
