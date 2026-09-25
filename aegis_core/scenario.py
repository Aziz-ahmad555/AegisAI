"""
Guided demo scenarios: scripted incidents played end to end through the real
modules. Only the *inputs* are scripted (sensor ramp, camera confidence, a
caller report); detection, fusion, proposals, the twin, routing and the
agents' briefing all run the normal code paths. Every event raised while a
scenario runs is tagged simulated, and the UI labels it SCENARIO / SIMULATED.
"""
import threading
import time
from dataclasses import dataclass

from . import coordinator
from . import events as ev
from .emergency_nlp import parse_emergency_report
from .events import Event

IDLE, RUNNING, PAUSED, FINISHED, STOPPED = "idle", "running", "paused", "finished", "stopped"

# Worded so offline keyword routing also consults all three agents
# ("fire", "trapped"/"rescue", "evacuation"/"route").
BRIEFING_QUESTION = ("Give the operator a short situation briefing: where the fire is and how confident the "
                     "evidence is, who is trapped or needs rescue, and which evacuation routes are blocked "
                     "or isolated.")


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    summary: str
    zone: str | None             # None -> the runner's default zone (the sensor zone)
    steps: tuple                 # (title, action, params); params may use {zone} / {zone_words}


SCENARIOS = {s.id: s for s in (
    Scenario(
        "corridor_fire", "Corridor fire",
        "Sensors rise, the camera confirms smoke and a caller reports two people trapped; "
        "the fire is declared and routes recompute around it.",
        None,
        (("Sensor readings rise", "sensors", {}),
         ("Camera confirms smoke", "camera", {"fire": 0.64, "smoke": 0.83}),
         ("Caller report arrives", "report",
          {"text": "Heavy smoke and flames in {zone_words}, two people are trapped and can't get out."}),
         ("Declare fire?", "declare", {}),
         ("Routes recompute", "routes", {}),
         ("Agents brief the operator", "briefing", {"question": BRIEFING_QUESTION}))),
    Scenario(
        "kitchen_fire", "Kitchen fire",
        "A cooking fire in the staff kitchen (Room102). The camera watching the kitchen sees it first; "
        "the monitored sensors are in another zone, so they stay quiet.",
        "Room102",
        (("Camera sees flames", "camera", {"fire": 0.78, "smoke": 0.52}),
         ("Caller report arrives", "report",
          {"text": "Fire in the staff kitchen in Room 102, flames on the stove and smoke spreading. "
                   "Everyone got out."}),
         ("Declare fire?", "declare", {}),
         ("Routes recompute", "routes", {}),
         ("Agents brief the operator", "briefing", {"question": BRIEFING_QUESTION}))),
    Scenario(
        "blocked_stairwell", "Smoke-logged stairwell",
        "Smoke fills the stairwell. Once it's declared, the stairwell is closed to evacuation "
        "and every room's route is re-checked against it.",
        "Stairwell",
        (("Caller report arrives", "report",
          {"text": "Visible smoke filling the stairwell near the second floor, nobody can use the stairs."}),
         ("Declare fire?", "declare", {}),
         ("Routes recompute", "routes", {}),
         ("Agents brief the operator", "briefing",
          {"question": "Brief the operator: which evacuation routes changed because the stairwell is blocked "
                       "by fire and smoke, and is anyone trapped?"}))),
    Scenario(
        "medical_emergency", "Medical emergency",
        "A caller reports an unconscious person in Room201. There's no fire, so nothing is proposed "
        "for the twin; the agents give the responders' route.",
        "Room201",
        (("Caller report arrives", "report",
          {"text": "Someone collapsed and is unconscious in Room 201, they are breathing but not responding."}),
         ("No automatic action", "no_action", {}),
         ("Responder route", "responder_route", {}),
         ("Agents brief the operator", "briefing",
          {"question": "Brief the operator: who is injured and needs medical help, where, "
                       "and which route should responders use?"}))),
)}
DEFAULT_SCENARIO = "corridor_fire"


class _Halt(Exception):
    """Raised inside the worker when the scenario is stopped or reset."""


class _End(Exception):
    """Raised by a step to finish the scenario early (e.g. operator dismissed)."""


class ScenarioRunner:
    def __init__(self, system, llm_provider=lambda: None, on_state=lambda state: None,
                 zone=None, countdown=10, time_scale=1.0):
        """
        llm_provider: callable returning the chat LLM (or None -> offline
            briefing); the coordinator falls back offline on any failure, so a
            demo never depends on the LLM being reachable.
        on_state: called with a state dict whenever progress changes.
        zone: the default scenario's zone (defaults to the sensor zone).
        time_scale: multiplies every wait (tests use 0 for instant runs).
        """
        self.system = system
        self.llm_provider = llm_provider
        self.on_state = on_state
        self.default_zone = zone or system.sensor_zone()
        self.countdown_total = countdown
        self.time_scale = time_scale
        self._lock = threading.RLock()
        self._resume = threading.Event()
        self._resume.set()
        self._thread = None
        self._halt = False
        self.scenario = SCENARIOS[DEFAULT_SCENARIO]
        self.zone = self.default_zone
        self._reset_state()

    @staticmethod
    def catalog():
        return [{"id": s.id, "title": s.title, "summary": s.summary} for s in SCENARIOS.values()]

    # ----- state ---------------------------------------------------------------

    @property
    def STEPS(self):                 # noqa: N802 - kept for callers that read the step titles
        return [title for title, _, _ in self.scenario.steps]

    def _reset_state(self):
        self.status = IDLE
        self.step = -1
        self.countdown = None
        self.briefing = None
        self.report = None
        self.proposal_id = None
        self.note = None
        self._routes_before = {}

    def state(self):
        with self._lock:
            steps = self.STEPS
            return {
                "status": self.status,
                "scenario": self.scenario.id,
                "title": self.scenario.title,
                "zone": self.zone,
                "step": self.step,
                "steps": [
                    {"title": t, "state": "done" if i < self.step or (i == self.step and self.status == FINISHED)
                     else "active" if i == self.step else "todo"}
                    for i, t in enumerate(steps)
                ],
                "countdown": self.countdown,
                "report": self.report,
                "briefing": self.briefing,
                "note": self.note,
                "active": self.status in (RUNNING, PAUSED, FINISHED, STOPPED),
            }

    def _publish_state(self):
        self.on_state(self.state())

    def _narrate(self, message, severity="info", **data):
        self.system.bus.publish(Event(ev.SCENARIO, "scenario", message, severity=severity,
                                      zone=data.pop("zone", self.zone), data=data))

    # ----- controls --------------------------------------------------------------

    def start(self, scenario_id=None):
        scenario = SCENARIOS.get(scenario_id or DEFAULT_SCENARIO)
        if scenario is None:
            return False
        with self._lock:
            if self.status in (RUNNING, PAUSED):
                return False
            self.system.reset_to_normal(source="scenario")
            self._reset_state()
            self.scenario = scenario
            self.zone = scenario.zone or self.default_zone
            self._halt = False
            self._resume.set()
            self.status = RUNNING
            self.system.bus.simulated = True
            self._thread = threading.Thread(target=self._run, name="scenario", daemon=True)
            self._thread.start()
        self._publish_state()
        return True

    def pause(self):
        with self._lock:
            if self.status != RUNNING:
                return False
            self.status = PAUSED
            self._resume.clear()
        self._narrate("Scenario paused by operator")
        self._publish_state()
        return True

    def resume(self):
        with self._lock:
            if self.status != PAUSED:
                return False
            self.status = RUNNING
            self._resume.set()
        self._narrate("Scenario resumed")
        self._publish_state()
        return True

    def stop(self):
        """Halt the run and leave the current state visible for inspection."""
        with self._lock:
            if self.status not in (RUNNING, PAUSED):
                return False
            self._halt = True
            self.status = STOPPED
            self._resume.set()
        self._join()
        self._narrate("Scenario stopped by operator - state left as-is; Reset returns to normal")
        self._publish_state()
        return True

    def reset(self):
        """Halt (if running) and return everything to normal."""
        with self._lock:
            self._halt = True
            self._resume.set()
        self._join()
        self.system.reset_to_normal(source="scenario")
        self.system.bus.simulated = False
        with self._lock:
            self._reset_state()
        self._publish_state()
        return True

    def _join(self):
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5)

    # ----- pacing -------------------------------------------------------------------

    def _check(self):
        if self._halt:
            raise _Halt()
        self._resume.wait()          # blocks while paused
        if self._halt:
            raise _Halt()

    def _sleep(self, seconds):
        end = time.monotonic() + seconds * self.time_scale
        while True:
            self._check()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _wait_for(self, condition, timeout):
        """Wait (pausable) until condition() is true; False on timeout."""
        deadline = time.monotonic() + timeout * max(self.time_scale, 0.05)
        while time.monotonic() < deadline:
            self._check()
            if condition():
                return True
            time.sleep(0.05)
        return condition()

    def _enter(self, index):
        with self._lock:
            self.step = index
        self._narrate(f"Step {index + 1}/{len(self.STEPS)}: {self.STEPS[index]}")
        self._publish_state()

    # ----- the script ------------------------------------------------------------------

    def _run(self):
        try:
            self._narrate(f"Guided scenario started - {self.scenario.title.lower()} in {self.zone}. "
                          "All data is simulated.", severity="warning")
            self._routes_before = self._routes()
            fmt = {"zone": self.zone, "zone_words": self.zone.replace("Corridor", "Corridor ")}
            for index, (_, action, params) in enumerate(self.scenario.steps):
                self._enter(index)
                params = {k: v.format(**fmt) if isinstance(v, str) else v for k, v in params.items()}
                getattr(self, "_step_" + action)(**params)
            with self._lock:
                self.status = FINISHED
            self._narrate("Scenario complete. Press Reset to return everything to normal.", severity="warning")
            self._publish_state()
        except _End:
            with self._lock:
                self.status = FINISHED
            self._narrate(self.note)
            self._publish_state()
        except _Halt:
            return

    # ----- steps: each drives one real module with scripted input -----------------------

    def _step_sensors(self):
        """The real sensor simulator ramps towards fire-like readings."""
        s = self.system
        s.sensors.trigger_event()
        self._wait_for(lambda: s.sensors.get_snapshot()["risk_level"] in ("ELEVATED", "HIGH", "CRITICAL"), 20)
        self._sleep(2)

    def _step_camera(self, fire, smoke):
        """Scripted camera confidence feeds the real fusion pipeline."""
        s = self.system
        s.set_camera_zone(self.zone)
        s.simulated_camera = (fire, smoke)
        self._wait_for(lambda: s.bus.latest(ev.CAMERA_FIRE, self.zone) is not None, 10)
        self._sleep(2)

    def _step_report(self, text):
        """A caller report through the real NLP + report ingestion."""
        parsed = parse_emergency_report(text)
        self.system.ingest_report(parsed)
        with self._lock:
            self.report = {"text": text, "event_types": parsed["event_types"],
                           "severity": parsed["severity"], "people_count": parsed["people_count"]}
        self._publish_state()
        self._sleep(2)

    def _step_declare(self):
        """The system's own proposal; the operator may confirm/dismiss, else auto-confirm."""
        s, zone = self.system, self.zone
        proposal = next((p for p in s.pending_actions()
                         if p["action"] == "declare_fire" and p["zone"] == zone), None)
        if proposal is None:
            proposal = s.propose("declare_fire", zone, f"Evidence of fire in {zone} - declare fire?",
                                 source="scenario")
        with self._lock:
            self.proposal_id = proposal["id"]
        for remaining in range(self.countdown_total, 0, -1):
            if not any(p["id"] == proposal["id"] for p in s.pending_actions()):
                break                               # the operator decided
            with self._lock:
                self.countdown = remaining
            self._publish_state()
            self._sleep(1)
        with self._lock:
            self.countdown = None
        if any(p["id"] == proposal["id"] for p in s.pending_actions()):
            s.decide(proposal["id"], True, decided_by="scenario auto-confirm")
        elif s.twin.zone_status.get(zone) != "FIRE":
            with self._lock:
                self.note = "Operator dismissed the fire declaration - scenario ended there."
            raise _End()
        # Close any other proposal the same evidence raised for this zone.
        for p in s.pending_actions():
            if p["zone"] == zone:
                s.decide(p["id"], False, decided_by="scenario")
        self._sleep(1)

    def _routes(self):
        rooms = [z for z in self.system.zones() if z.startswith("Room")]
        return {room: self.system.twin.compute_evacuation_route(room) for room in rooms}

    def _step_routes(self):
        """Real routing over the updated twin, compared with the routes at the start."""
        s, zone = self.system, self.zone
        routes = self._routes()
        isolated = [room for room, r in routes.items() if r["path"] is None]
        via_hazard = [room for room, r in routes.items() if r["path"] and r["through_hazard"]]
        for room in isolated:
            s.bus.publish(Event(ev.ROUTE_STATUS, "twin", f"{room} is isolated - no route to any exit, rescue required",
                                severity="critical", zone=room))
        rerouted = []
        for room, r in routes.items():
            before = self._routes_before.get(room, {}).get("path")
            if r["path"] and before and r["path"] != before:
                rerouted.append(room)
                s.bus.publish(Event(ev.ROUTE_STATUS, "twin",
                                    f"{room} rerouted: {' -> '.join(r['path'])} (was {' -> '.join(before)})",
                                    severity="warning", zone=room))
        open_rooms = [r for r in routes if r not in isolated]
        detail = f"{len(open_rooms)} of {len(routes)} rooms can still evacuate"
        if not isolated and not rerouted:
            detail += f"; no room's route passed through {zone}"
        s.bus.publish(Event(ev.ROUTE_STATUS, "twin", f"Routes recomputed around {zone}: {detail}",
                            severity="warning" if isolated else "info", zone=zone,
                            data={"isolated": isolated, "through_hazard": via_hazard, "rerouted": rerouted}))
        self._publish_state()
        self._sleep(2)

    def _step_no_action(self):
        """Medical-only report: confirm nothing was proposed for the twin."""
        pending = [p for p in self.system.pending_actions() if p["zone"] == self.zone]
        if pending:
            self._narrate(f"{len(pending)} action(s) proposed for {self.zone} - awaiting the operator")
        else:
            self._narrate(f"No fire evidence in {self.zone}, so no twin action is proposed. "
                          "Dispatching medical responders is a human decision.")
        self._publish_state()
        self._sleep(2)

    def _step_responder_route(self):
        """The route responders use to reach the patient (the room's exit route, reversed)."""
        r = self.system.twin.compute_evacuation_route(self.zone)
        if r["path"] is None:
            message, severity = f"No route from any exit to {self.zone} - responders cannot reach it", "critical"
        else:
            status = "passes a hazard zone" if r["through_hazard"] else "clear"
            message = f"Responder access to {self.zone}: {' -> '.join(reversed(r['path']))} ({status})"
            severity = "warning" if r["through_hazard"] else "info"
        self.system.bus.publish(Event(ev.ROUTE_STATUS, "twin", message, severity=severity, zone=self.zone))
        self._publish_state()
        self._sleep(2)

    def _step_briefing(self, question):
        """The agents' briefing (LLM if available, offline routing otherwise)."""
        briefing = coordinator.ask_coordinator(question, self.llm_provider())
        self._check()
        with self._lock:
            self.briefing = briefing
        self.system.bus.publish(Event(ev.BRIEFING, "agents", "Agents posted a situation briefing",
                                      severity="info", zone=self.zone))
