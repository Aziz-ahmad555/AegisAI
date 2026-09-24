"""
Guided demo scenario: a scripted incident played end to end through the real
modules. Only the *inputs* are scripted (sensor ramp, camera confidence, a
caller report); detection, fusion, proposals, the twin, routing and the
agents' briefing all run the normal code paths. Every event raised while a
scenario runs is tagged simulated, and the UI labels it SCENARIO / SIMULATED.
"""
import threading
import time

from . import coordinator
from . import events as ev
from .emergency_nlp import parse_emergency_report
from .events import Event

IDLE, RUNNING, PAUSED, FINISHED, STOPPED = "idle", "running", "paused", "finished", "stopped"

BRIEFING_QUESTION = ("Give the operator a short situation briefing: where the fire is, how confident the "
                     "evidence is, who is reported at risk, and which evacuation routes are blocked or isolated.")


class _Halt(Exception):
    """Raised inside the worker when the scenario is stopped or reset."""


class ScenarioRunner:
    STEPS = [
        "Sensor readings rise",
        "Camera confirms smoke",
        "Caller report arrives",
        "Declare fire?",
        "Routes recompute",
        "Agents brief the operator",
    ]

    def __init__(self, system, llm_provider=lambda: None, on_state=lambda state: None,
                 zone=None, countdown=10, time_scale=1.0):
        """
        llm_provider: callable returning the chat LLM (or None -> offline
            briefing); the coordinator falls back offline on any failure, so a
            demo never depends on the LLM being reachable.
        on_state: called with a state dict whenever progress changes.
        time_scale: multiplies every wait (tests use 0 for instant runs).
        """
        self.system = system
        self.llm_provider = llm_provider
        self.on_state = on_state
        self.zone = zone or system.sensor_zone()
        self.countdown_total = countdown
        self.time_scale = time_scale
        self._lock = threading.RLock()
        self._resume = threading.Event()
        self._resume.set()
        self._thread = None
        self._halt = False
        self._reset_state()

    # ----- state ---------------------------------------------------------------

    def _reset_state(self):
        self.status = IDLE
        self.step = -1
        self.countdown = None
        self.briefing = None
        self.report = None
        self.proposal_id = None
        self.note = None

    def state(self):
        with self._lock:
            return {
                "status": self.status,
                "zone": self.zone,
                "step": self.step,
                "steps": [
                    {"title": t, "state": "done" if i < self.step or (i == self.step and self.status == FINISHED)
                     else "active" if i == self.step else "todo"}
                    for i, t in enumerate(self.STEPS)
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

    def start(self):
        with self._lock:
            if self.status in (RUNNING, PAUSED):
                return False
            self.system.reset_to_normal(source="scenario")
            self._reset_state()
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
        s, zone = self.system, self.zone
        try:
            self._narrate(f"Guided scenario started - scripted fire incident in {zone}. All data is simulated.",
                          severity="warning")

            # 1. Real sensor simulator ramps towards fire-like readings.
            self._enter(0)
            s.sensors.trigger_event()
            self._wait_for(lambda: s.sensors.get_snapshot()["risk_level"] in ("ELEVATED", "HIGH", "CRITICAL"), 20)
            self._sleep(2)

            # 2. Scripted camera confidence feeds the real fusion pipeline.
            self._enter(1)
            s.set_camera_zone(zone)
            s.simulated_camera = (0.64, 0.83)            # (fire, smoke)
            self._wait_for(lambda: s.bus.latest(ev.CAMERA_FIRE, zone) is not None, 10)
            self._sleep(2)

            # 3. A caller report through the real NLP + report ingestion.
            self._enter(2)
            text = (f"Heavy smoke and flames in {zone.replace('Corridor', 'Corridor ')}, "
                    "two people are trapped and can't get out.")
            parsed = parse_emergency_report(text)
            s.ingest_report(parsed)
            with self._lock:
                self.report = {"text": text, "event_types": parsed["event_types"],
                               "severity": parsed["severity"], "people_count": parsed["people_count"]}
            self._publish_state()
            self._sleep(2)

            # 4. The system's own proposal; operator may confirm/dismiss, else auto-confirm.
            self._enter(3)
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
                    self.status = FINISHED
                    self.note = "Operator dismissed the fire declaration - scenario ended there."
                self._narrate(self.note)
                self._publish_state()
                return
            # Close any other proposal the same evidence raised for this zone.
            for p in s.pending_actions():
                if p["zone"] == zone:
                    s.decide(p["id"], False, decided_by="scenario")
            self._sleep(1)

            # 5. Real routing over the updated twin.
            self._enter(4)
            rooms = [z for z in s.zones() if z.startswith("Room")]
            isolated, via_hazard = [], []
            for room in rooms:
                r = s.twin.compute_evacuation_route(room)
                if r["path"] is None:
                    isolated.append(room)
                elif r["through_hazard"]:
                    via_hazard.append(room)
            for room in isolated:
                s.bus.publish(Event(ev.ROUTE_STATUS, "twin", f"{room} is isolated - no route to any exit, rescue required",
                                    severity="critical", zone=room))
            open_rooms = [r for r in rooms if r not in isolated]
            s.bus.publish(Event(ev.ROUTE_STATUS, "twin",
                                f"Routes recomputed around {zone}: {len(open_rooms)} of {len(rooms)} rooms can still evacuate",
                                severity="warning" if isolated else "info", zone=zone,
                                data={"isolated": isolated, "through_hazard": via_hazard}))
            self._publish_state()
            self._sleep(2)

            # 6. The agents' briefing (LLM if available, offline routing otherwise).
            self._enter(5)
            briefing = coordinator.ask_coordinator(BRIEFING_QUESTION, self.llm_provider())
            self._check()
            with self._lock:
                self.briefing = briefing
            s.bus.publish(Event(ev.BRIEFING, "agents", "Agents posted a situation briefing", severity="info", zone=zone))
            with self._lock:
                self.status = FINISHED
            self._narrate("Scenario complete. Press Reset to return everything to normal.", severity="warning")
            self._publish_state()
        except _Halt:
            return
