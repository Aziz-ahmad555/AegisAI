"""App-level wiring: auth on HTTP + WebSocket, report -> proposal -> twin over
the real Flask/Socket.IO stack, live agents answering through /api/chat."""
import pytest

import app as aegis
from helpers import api_post
from helpers import login as _login


def login(c):
    r = _login(c)
    assert r.status_code == 302


@pytest.mark.parametrize("path", ["/", "/twin", "/sensors", "/nlp", "/chat", "/aerial", "/vision", "/api/timeline"])
def test_pages_require_login(client, path):
    r = client.get(path)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_bad_password_is_rejected(client):
    r = _login(client, password="wrong")
    assert r.status_code == 401 and b"Invalid credentials" in r.data


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

    r = api_post(client, "/api/analyze_report", {"text": "There is a fire in Corridor B, two people trapped"})
    body = r.get_json()
    assert body["zones"] == ["CorridorB"]
    assert body["proposals"] and aegis.twin.zone_status["CorridorB"] == "SAFE"

    received = sio.get_received()
    assert any(m["name"] == "timeline_event" and m["args"][0]["type"] == "ACTION_PROPOSED" for m in received)

    sio.emit("decide_action", {"id": body["proposals"][0]["id"], "approve": True})
    assert aegis.twin.zone_status["CorridorB"] == "FIRE"

    answer = api_post(client, "/api/chat", {"question": "Which evacuation routes are blocked?"}).get_json()
    assert answer["mode"] == "offline"
    assert "CorridorB" in answer["answer"]
    sio.disconnect()


def test_route_reply_goes_only_to_requester(client):
    login(client)
    a = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    b = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    a.get_received()
    b.get_received()
    a.emit("request_route", {"start": "Room101"})
    assert any(m["name"] == "route_result" for m in a.get_received())
    assert not any(m["name"] == "route_result" for m in b.get_received())
    a.disconnect()
    b.disconnect()


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
    r = api_post(client, "/api/actions/99999", {"approve": True})
    assert r.status_code == 404


# --- streaming chat over Socket.IO -------------------------------------------------------

def _collect_chat(sio, timeout=10.0):
    """Gather chat_event messages until the server sends 'done'."""
    import time

    events, deadline = [], time.time() + timeout
    while time.time() < deadline:
        for m in sio.get_received():
            if m["name"] == "chat_event":
                events.append(m["args"][0])
                if events[-1]["type"] == "done":
                    return events
        time.sleep(0.02)
    raise AssertionError(f"no 'done' event; got {events}")


def test_chat_streams_to_the_asker_and_remembers_the_conversation(client, monkeypatch):
    from fakes import claude, text, tool

    llm = claude([
        ("tool_use", [tool("consult_route_agent")]),
        ("end_turn", [text("All routes are open.")]),
        ("end_turn", [text("Room202 exits via Stairwell.")]),
    ])
    monkeypatch.setattr(aegis, "llm_client", llm)
    fake = llm.client
    login(client)
    asker = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    other = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    asker.get_received()
    other.get_received()

    asker.emit("chat_ask", {"question": "Are the routes open?"})
    events = _collect_chat(asker)
    assert {"type": "agent", "agent": "Route"} in events
    assert [e for e in events if e["type"] == "text"] == []          # raw model text never sent
    assert [e for e in events if e["type"] == "html"][-1]["html"].strip() == "<p>All routes are open.</p>"
    assert not any(m["name"] == "chat_event" for m in other.get_received())   # nobody else sees it

    asker.emit("chat_ask", {"question": "And from Room202?"})
    _collect_chat(asker)
    followup = fake.requests[-1]["messages"]
    assert followup[:2] == [{"role": "user", "content": "Are the routes open?"},
                            {"role": "assistant", "content": "All routes are open."}]

    assert api_post(client, "/api/chat/reset").status_code == 200
    asker.emit("chat_ask", {"question": "Fresh start?"})
    _collect_chat(asker)
    assert fake.requests[-1]["messages"] == [{"role": "user", "content": "Fresh start?"}]
    asker.disconnect()
    other.disconnect()


def test_chat_forwards_evidence_over_the_socket_and_api(client, monkeypatch):
    from fakes import claude, text, tool

    monkeypatch.setattr(aegis, "llm_client", claude([
        ("tool_use", [tool("consult_fire_agent")]),
        ("end_turn", [text("No fire declared.")]),
    ]))
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.get_received()
    sio.emit("chat_ask", {"question": "Is there a fire?"})
    events = _collect_chat(sio)
    evidence = [e for e in events if e["type"] == "evidence"]
    assert [e["agent"] for e in evidence] == ["Fire"]
    assert {"source", "zone", "value", "time"} == set(evidence[0]["items"][0])
    sio.disconnect()

    monkeypatch.setattr(aegis, "llm_client", None)       # offline: /api/chat returns evidence too
    body = api_post(client, "/api/chat", {"question": "Is there a fire?"}).get_json()
    assert [e["agent"] for e in body["evidence"]] == ["Fire"]
    assert body["evidence"][0]["items"][0]["value"] == "no declared fires"


def test_chat_socket_rejects_oversized_question(client):
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.get_received()
    sio.emit("chat_ask", {"question": "x" * (aegis.security.MAX_QUESTION_CHARS + 1)})
    events = [m["args"][0] for m in sio.get_received() if m["name"] == "chat_event"]
    assert events and events[0]["type"] == "error" and "too long" in events[0]["text"]
    sio.disconnect()


def test_chat_page_shows_the_configured_model(client, monkeypatch):
    from fakes import claude

    monkeypatch.setattr(aegis, "llm_client", claude([("end_turn", [])]))
    login(client)
    assert b"claude-sonnet-5" in client.get("/chat").data


def test_malicious_model_output_reaches_the_browser_only_as_inert_html(client, monkeypatch):
    from fakes import claude, text

    evil = ("**Warning** <img src=x onerror=alert(1)> <script>alert(2)</script> "
            "[x](javascript:alert(3)) ![p](https://attacker.example/leak)")
    monkeypatch.setattr(aegis, "llm_client", claude([("end_turn", [text(evil)])]))
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.get_received()
    sio.emit("chat_ask", {"question": "status?"})
    events = _collect_chat(sio)
    final = [e for e in events if e["type"] == "html"][-1]["html"]
    assert "<strong>Warning</strong>" in final
    from test_safe_markdown import parsed

    for needle in ("<img", "<script", "attacker.example"):
        assert needle not in final
    for _tag, attrs in parsed(final):         # no handler / script scheme in any real tag
        assert not any(k.startswith("on") or "javascript:" in (v or "").lower() for k, v in attrs.items())
    assert "&lt;img src=x onerror=alert(1)&gt;" in final                  # visible, not executable
    sio.disconnect()


def test_successful_groq_stream_is_not_replaced_by_the_offline_answer(client, monkeypatch):
    # Regression guard: a working Groq answer (with a tool round) must reach
    # the browser as the model's answer - never the offline fallback text.
    from fakes import groq, groq_text, groq_tool_calls

    llm = groq([
        groq_tool_calls(("c1", "consult_fire_agent", "{}")),
        groq_text("**No active fires.** All zones are clear."),
        groq_text("Still no fires."),
    ])
    monkeypatch.setattr(aegis, "llm_client", llm)
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.get_received()

    turns = []
    for question, expected in [("is there any fire right now?", "<strong>No active fires.</strong>"),
                               ("and now?", "Still no fires.")]:          # 2nd turn carries history
        sio.emit("chat_ask", {"question": question})
        events = _collect_chat(sio)
        turns.append(events)
        final = [e for e in events if e["type"] == "html"][-1]["html"]
        assert expected in final
        assert "OFFLINE MODE" not in final
        assert not [e for e in events if e["type"] in ("notice", "reset", "error")], events
        assert {"type": "mode", "mode": "llm", "provider": "groq", "model": llm.model} in events
    assert {"type": "agent", "agent": "Fire"} in turns[0]                    # tool round happened
    second_request = llm.client.requests[-1]["messages"]
    assert {"role": "user", "content": "is there any fire right now?"} in second_request
    sio.disconnect()


# --- guided scenario over Socket.IO ---------------------------------------------------------

def _scenario_states(sio, until_status, timeout=15.0):
    import time

    states, deadline = [], time.time() + timeout
    while time.time() < deadline:
        for m in sio.get_received():
            if m["name"] == "scenario_state":
                states.append(m["args"][0])
                if states[-1]["status"] == until_status:
                    return states
        time.sleep(0.02)
    raise AssertionError(f"scenario never reached {until_status}: {[s['status'] for s in states]}")


def test_scenario_runs_over_the_socket_and_briefing_arrives_sanitized(client, monkeypatch):
    from fakes import groq, groq_text

    evil = "**Fire in CorridorB.** <img src=x onerror=alert(1)> ![p](https://attacker.example/x)"
    monkeypatch.setattr(aegis, "llm_client", groq([groq_text(evil)]))
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    first = [m["args"][0] for m in sio.get_received() if m["name"] == "scenario_state"]
    assert first and first[0]["status"] == "idle" and first[0]["active"] is False

    sio.emit("scenario_command", {"action": "start"})
    states = _scenario_states(sio, "finished")
    final = states[-1]
    assert final["active"] and [s["state"] for s in final["steps"]] == ["done"] * 6
    assert "briefing" not in final                                    # raw model text never sent
    html = final["briefing_html"]
    assert "<strong>Fire in CorridorB.</strong>" in html
    assert "<img" not in html and "attacker.example" not in html
    assert aegis.twin.zone_status["CorridorB"] == "FIRE"

    sio.emit("scenario_command", {"action": "reset"})
    _scenario_states(sio, "idle")
    assert aegis.twin.zone_status["CorridorB"] == "SAFE" and aegis.system.pending_actions() == []
    sio.disconnect()


def test_scenario_commands_require_login_and_ignore_unknown_actions(client):
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    assert not sio.is_connected()                                       # refused before login
    login(client)
    sio = aegis.socketio.test_client(aegis.app, flask_test_client=client)
    sio.emit("scenario_command", {"action": "rm -rf"})
    assert aegis.scenario.status == "idle"
    sio.disconnect()


def test_every_page_offers_the_scenario_controls(client):
    login(client)
    for path in ["/", "/twin", "/chat", "/nlp"]:
        html = client.get(path).get_data(as_text=True)
        assert 'id="scenario-run"' in html and 'id="scenario-banner"' in html and "SCENARIO &middot; SIMULATED" in html


def test_healthz_is_public_and_minimal(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.get_json() == {"status": "ok", "zones": 10}
