import os
import threading
import time
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, disconnect
from building_state import BuildingDigitalTwin

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("AEGISAI_SECRET_KEY", "aegisai-digital-twin-demo-key-change-in-production")
socketio = SocketIO(app, cors_allowed_origins="*")

twin = BuildingDigitalTwin()

# In a real deployment this would be a proper user database with hashed
# passwords (e.g. via werkzeug.security or a dedicated auth provider) and
# role-based access control (operators vs. admins). For this portfolio
# demo, a single operator credential set via environment variable
# demonstrates the pattern without overbuilding infrastructure that would
# never be exercised by a single-user local demo.
OPERATOR_USERNAME = os.environ.get("AEGISAI_USERNAME", "operator")
OPERATOR_PASSWORD = os.environ.get("AEGISAI_PASSWORD", "aegisai2026")


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
            return redirect(url_for("index"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("dashboard.html")


def broadcast_state():
    snapshot = twin.get_state_snapshot()
    socketio.emit("state_update", snapshot)


@socketio.on("connect")
def handle_connect():
    # Reject socket connections from anyone who hasn't logged in via the
    # HTTP session first - prevents bypassing the login page by talking
    # to the WebSocket endpoint directly.
    if not session.get("logged_in"):
        print("Rejected unauthenticated socket connection")
        disconnect()
        return
    print("Authenticated client connected")
    broadcast_state()


@socketio.on("trigger_fire")
def handle_trigger_fire(data):
    if not session.get("logged_in"):
        disconnect()
        return
    zone = data.get("zone")
    if zone:
        twin.start_fire(zone)
        broadcast_state()


@socketio.on("clear_zone")
def handle_clear_zone(data):
    if not session.get("logged_in"):
        disconnect()
        return
    zone = data.get("zone")
    if zone:
        twin.clear_zone(zone)
        broadcast_state()


@socketio.on("request_route")
def handle_request_route(data):
    if not session.get("logged_in"):
        disconnect()
        return
    start = data.get("start", "Room101")
    route = twin.compute_evacuation_route(start)
    socketio.emit("route_result", {"start": start, "route": route})


def periodic_broadcast():
    while True:
        time.sleep(2)
        broadcast_state()


if __name__ == "__main__":
    broadcaster = threading.Thread(target=periodic_broadcast, daemon=True)
    broadcaster.start()

    print("AegisAI Phase 10 - Digital Twin Dashboard (secured)")
    print(f"Login with username '{OPERATOR_USERNAME}' (set AEGISAI_USERNAME/AEGISAI_PASSWORD env vars to change)")
    print("Open http://127.0.0.1:5000 in your browser")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
