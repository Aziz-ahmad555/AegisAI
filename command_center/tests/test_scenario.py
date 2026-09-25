"""Guided demo scenario: plays end to end through the real modules, labels
everything simulated, respects operator decisions, and resets cleanly."""
import time

import pytest

from aegis_core import coordinator
from aegis_core import events as ev
from aegis_core.building_state import BuildingDigitalTwin
from aegis_core.scenario import FINISHED, IDLE, PAUSED, RUNNING, STOPPED, ScenarioRunner
from aegis_core.sensor_state import SensorFusionState
from aegis_core.system import AegisSystem

from fakes import groq, groq_text


@pytest.fixture
def system():
    s = AegisSystem(BuildingDigitalTwin(), SensorFusionState())
    coordinator.init_agents(s)
    return s


def make(system, **kw):
    states = []
    kw.setdefault("time_scale", 0)
    runner = ScenarioRunner(system, on_state=states.append, **kw)
    return runner, states


def wait_until(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def types(system):
    return [e["type"] for e in system.bus.timeline(limit=200)]


def test_full_run_offline_plays_every_step_through_the_real_modules(system):
    runner, states = make(system, zone="CorridorB")
    assert runner.start()
    wait_until(lambda: runner.status == FINISHED)

    assert system.twin.zone_status["CorridorB"] == "FIRE"          # twin marked by the confirmation
    assert system.pending_actions() == []                           # nothing left dangling
    assert len(system.reports()) == 1                               # caller report went through NLP
    msgs = [e["message"] for e in system.bus.timeline(limit=200)]
    assert any("Room201 is isolated" in m for m in msgs)            # routing recomputed
    assert any("confirmed" in m.lower() for m in msgs)
    assert ev.BRIEFING in types(system)
    assert runner.briefing.startswith("[OFFLINE MODE")              # works with no LLM at all
    for agent in ("FireAgent", "MedicalAgent", "RouteAgent"):        # offline briefing covers all three
        assert agent in runner.briefing
    assert "2 people" in runner.briefing                            # the caller's trapped people
    final = states[-1]
    assert final["status"] == FINISHED and all(s["state"] == "done" for s in final["steps"])


def test_every_event_during_the_scenario_is_labelled_simulated(system):
    runner, _ = make(system)
    runner.start()
    wait_until(lambda: runner.status == FINISHED)
    scenario_events = [e for e in system.bus.timeline(limit=200) if e["type"] != ev.RESET]
    assert scenario_events and all(e["data"].get("simulated") for e in scenario_events)


def test_events_are_not_simulated_outside_a_scenario(system):
    system.start_fire("Room101")
    assert not system.bus.timeline()[0]["data"].get("simulated")


def test_operator_can_confirm_during_the_countdown(system):
    runner, _ = make(system, countdown=500, time_scale=0.01)      # countdown would take 5 s
    runner.start()
    wait_until(lambda: runner.step == 3 and runner.countdown is not None)
    pid = runner.proposal_id
    system.decide(pid, True)                                       # operator clicks Confirm
    wait_until(lambda: runner.status == FINISHED)
    assert system.twin.zone_status[runner.zone] == "FIRE"
    confirms = [e for e in system.bus.timeline(limit=200) if e["type"] == ev.ACTION_CONFIRMED]
    assert confirms[-1]["source"] == "operator"                   # not auto-confirmed


def test_operator_dismissal_ends_the_scenario_without_declaring_fire(system):
    runner, _ = make(system, countdown=500, time_scale=0.01)
    runner.start()
    wait_until(lambda: runner.step == 3 and runner.countdown is not None)
    system.decide(runner.proposal_id, False)
    wait_until(lambda: runner.status == FINISHED)
    assert system.twin.zone_status[runner.zone] == "SAFE"
    assert "dismissed" in runner.note


def test_auto_confirm_is_attributed_to_the_scenario(system):
    runner, _ = make(system)
    runner.start()
    wait_until(lambda: runner.status == FINISHED)
    confirm = next(e for e in system.bus.timeline(limit=200) if e["type"] == ev.ACTION_CONFIRMED)
    assert confirm["source"] == "scenario auto-confirm"


def test_pause_holds_progress_and_resume_continues(system):
    runner, _ = make(system, countdown=3, time_scale=0.05)
    runner.start()
    wait_until(lambda: runner.step >= 1)
    assert runner.pause() and runner.status == PAUSED
    held = runner.step
    time.sleep(0.6)
    assert runner.step == held                                      # frozen while paused
    assert runner.resume() and runner.status == RUNNING
    wait_until(lambda: runner.status == FINISHED, timeout=20)


def test_stop_leaves_state_and_reset_returns_everything_to_normal(system):
    runner, _ = make(system, countdown=500, time_scale=0.01)
    runner.start()
    wait_until(lambda: runner.step == 3)
    assert runner.stop() and runner.status == STOPPED
    assert system.pending_actions()                                 # left as-is for inspection

    runner.reset()
    assert runner.status == IDLE
    assert all(s != "FIRE" for s in system.twin.zone_status.values())
    assert system.pending_actions() == [] and system.reports() == []
    assert system.sensors.get_snapshot()["risk_level"] == "NORMAL"
    assert system.camera_confidence() == (0.0, 0.0)
    assert system.bus.simulated is False
    assert system.bus.timeline()[0]["type"] == ev.RESET


def test_reset_after_finish_clears_the_fire(system):
    runner, _ = make(system)
    runner.start()
    wait_until(lambda: runner.status == FINISHED)
    runner.reset()
    assert system.twin.zone_status[runner.zone] == "SAFE"
    assert system.twin.get_blocked_edges() == []


def test_cannot_start_twice_but_can_rerun_after_finishing(system):
    runner, _ = make(system, countdown=500, time_scale=0.01)
    assert runner.start()
    assert not runner.start()
    runner.reset()
    runner.time_scale, runner.countdown_total = 0, 1
    assert runner.start()
    wait_until(lambda: runner.status == FINISHED)


def test_briefing_uses_the_llm_when_available(system):
    llm = groq([groq_text("**Fire in CorridorB.** Room201 isolated.")])
    runner, _ = make(system, llm_provider=lambda: llm)
    runner.start()
    wait_until(lambda: runner.status == FINISHED)
    assert runner.briefing == "**Fire in CorridorB.** Room201 isolated."


# --- scenario library -----------------------------------------------------------------------

def test_catalog_lists_every_scenario():
    ids = [s["id"] for s in ScenarioRunner.catalog()]
    assert ids == ["corridor_fire", "kitchen_fire", "blocked_stairwell", "medical_emergency"]
    assert all(s["title"] and s["summary"] for s in ScenarioRunner.catalog())


def test_unknown_scenario_is_refused(system):
    runner, _ = make(system)
    assert not runner.start("delete_everything")
    assert runner.status == IDLE


def run_scenario(system, scenario_id):
    runner, states = make(system)
    assert runner.start(scenario_id)
    wait_until(lambda: runner.status == FINISHED)
    return runner, states, [e["message"] for e in system.bus.timeline(limit=200)]


def test_kitchen_fire_is_camera_led_and_declares_room102(system):
    runner, states, msgs = run_scenario(system, "kitchen_fire")
    assert runner.zone == "Room102" and states[-1]["title"] == "Kitchen fire"
    assert system.twin.zone_status["Room102"] == "FIRE"
    assert system.camera_zone() == "Room102"
    assert system.reports()[0]["zones"] == ["Room102"]
    assert system.sensors.get_snapshot()["risk_level"] == "NORMAL"      # sensors are elsewhere: not scripted
    assert system.pending_actions() == []
    assert any("Routes recomputed around Room102" in m for m in msgs)


def test_blocked_stairwell_declares_the_stairwell_and_rechecks_routes(system):
    runner, _, msgs = run_scenario(system, "blocked_stairwell")
    assert system.twin.zone_status["Stairwell"] == "FIRE"
    assert [z for z, s in system.twin.zone_status.items() if s == "FIRE"] == ["Stairwell"]
    route = next(m for m in msgs if m.startswith("Routes recomputed around Stairwell"))
    # Honest outcome for this building: every room still has an exit that avoids the stairwell.
    assert "5 of 5 rooms can still evacuate" in route
    assert "RouteAgent" in runner.briefing and "FireAgent" in runner.briefing


def test_routes_step_reports_rerouted_rooms(system):
    runner, _ = make(system)
    runner._routes_before = runner._routes()
    runner.zone = "ExitMain"
    system.start_fire("ExitMain")                  # Room202's shortest exit is gone
    runner._step_routes()
    msgs = [e["message"] for e in system.bus.timeline(limit=50)]
    assert "Room202 rerouted: Room202 -> Stairwell -> ExitEmergency (was Room202 -> ExitMain)" in msgs
    summary = next(m for m in msgs if m.startswith("Routes recomputed around ExitMain"))
    assert "no room's route passed through" not in summary


def test_medical_emergency_proposes_nothing_and_gives_the_responder_route(system):
    runner, _, msgs = run_scenario(system, "medical_emergency")
    assert all(s != "FIRE" for s in system.twin.zone_status.values())
    assert system.pending_actions() == []
    assert not [e for e in system.bus.timeline(limit=200) if e["type"] == ev.ACTION_PROPOSED]
    assert system.reports()[0]["event_types"] == ["MEDICAL"]
    assert any("no twin action is proposed" in m for m in msgs)
    assert any(m == "Responder access to Room201: ExitMain -> CorridorB -> Room201 (clear)" for m in msgs)
    assert "MedicalAgent" in runner.briefing and "RouteAgent" in runner.briefing
    assert "FireAgent" not in runner.briefing                            # no fire question asked


@pytest.mark.parametrize("scenario_id", ["kitchen_fire", "blocked_stairwell", "medical_emergency"])
def test_every_scenario_is_labelled_simulated_and_resets_cleanly(system, scenario_id):
    runner, _, _ = run_scenario(system, scenario_id)
    events = [e for e in system.bus.timeline(limit=200) if e["type"] != ev.RESET]
    assert events and all(e["data"].get("simulated") for e in events)
    runner.reset()
    assert all(s != "FIRE" for s in system.twin.zone_status.values())
    assert system.reports() == [] and system.pending_actions() == []
