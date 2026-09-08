import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from building_state import BuildingDigitalTwin

app = Flask(__name__)
app.config["SECRET_KEY"] = "aegisai-digital-twin-demo"
socketio = SocketIO(app, cors_allowed_origins="*")

twin = BuildingDigitalTwin()


def broadcast_state():
    """Push the current state to all connected clients."""
    snapshot = twin.get_state_snapshot()
    socketio.emit("state_update", snapshot)


@app.route("/")
def index():
    return render_template("dashboard.html")


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    broadcast_state()


@socketio.on("trigger_fire")
def handle_trigger_fire(data):
    zone = data.get("zone")
    if zone:
        twin.start_fire(zone)
        broadcast_state()


@socketio.on("clear_zone")
def handle_clear_zone(data):
    zone = data.get("zone")
    if zone:
        twin.clear_zone(zone)
        broadcast_state()


@socketio.on("request_route")
def handle_request_route(data):
    start = data.get("start", "Room101")
    route = twin.compute_evacuation_route(start)
    socketio.emit("route_result", {"start": start, "route": route})


def periodic_broadcast():
    """Keep clients in sync even without explicit actions (e.g. risk decay over time)."""
    while True:
        time.sleep(2)
        broadcast_state()


if __name__ == "__main__":
    broadcaster = threading.Thread(target=periodic_broadcast, daemon=True)
    broadcaster.start()

    print("AegisAI Phase 10 - Digital Twin Dashboard")
    print("Open http://127.0.0.1:5000 in your browser")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
