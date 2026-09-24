"""App-level wiring: auth on HTTP + WebSocket, report -> proposal -> twin over
the real Flask/Socket.IO stack, live agents answering through /api/chat."""
import pytest

import app as aegis


@pytest.fixture
def client():
    aegis.app.config["TESTING"] = True
    c = aegis.app.test_client()
    yield c
    for zone in aegis.system.zones():
        aegis.system.clear_zone(zone)
    for p in aegis.system.pending_actions():
        aegis.system.decide(p["id"], approve=False)


def login(c):
    r = c.post("/login", data={"username": aegis.OPERATOR_USERNAME, "password": aegis.OPERATOR_PASSWORD})
    assert r.status_code == 302


@pytest.mark.parametrize("path", ["/", "/twin", "/sensors", "/nlp", "/chat", "/aerial", "/vision", "/api/timeline"])
def test_pages_require_login(client, path):
    r = client.get(path)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_bad_password_is_rejected(client):
    r = client.post("/login", data={"username": aegis.OPERATOR_USERNAME, "password": "wrong"})
    assert r.status_code == 200 and b"Invalid credentials" in r.data


def test_unauthenticated_socket_is_refused(client):
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    assert not sio.is_connected()


def test_socket_connect_sends_initial_state_to_that_client(client):
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    names = {m["name"] for m in sio.get_received()}
    assert {"state_update", "sensor_update", "timeline", "pending_actions"} <= names
    sio.disconnect()


def test_report_to_confirmed_fire_end_to_end(client):
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.get_received()

    r = client.post("/api/analyze_report", json={"text": "There is a fire in Corridor B, two people trapped"})
    body = r.get_json()
    assert body["zones"] == ["CorridorB"]
    assert body["proposals"] and aegis.twin.zone_status["CorridorB"] == "SAFE"

    received = sio.get_received()
    assert any(m["name"] == "timeline_event" and m["args"][0]["type"] == "ACTION_PROPOSED" for m in received)

    sio.emit("decide_action", {"id": body["proposals"][0]["id"], "approve": True})
    assert aegis.twin.zone_status["CorridorB"] == "FIRE"

    answer = client.post("/api/chat", json={"question": "Which evacuation routes are blocked?"}).get_json()
    assert answer["mode"] == "offline"
    assert "CorridorB" in answer["answer"]
    sio.disconnect()


def test_route_reply_goes_only_to_requester(client):
    login(client)
    a = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    b = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    a.get_received(); b.get_received()
    a.emit("request_route", {"start": "Room101"})
    assert any(m["name"] == "route_result" for m in a.get_received())
    assert not any(m["name"] == "route_result" for m in b.get_received())
    a.disconnect(); b.disconnect()


def test_invalid_socket_input_is_ignored(client):
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.emit("trigger_fire", {"zone": "Nowhere"})
    sio.emit("request_route", {"start": "Nowhere"})
    sio.emit("decide_action", {"id": "not-a-number"})
    assert "Nowhere" not in aegis.twin.zone_status
    sio.disconnect()


def test_decide_unknown_action_returns_404(client):
    login(client)
    r = client.post("/api/actions/99999", json={"approve": True})
    assert r.status_code == 404
