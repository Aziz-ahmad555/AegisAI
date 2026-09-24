"""Stage 2 integration: modules connected through AegisSystem + EventBus."""
import pytest

import events as ev
from agents import FireAgent, MedicalAgent, RouteAgent
from building_state import BuildingDigitalTwin
from emergency_nlp import parse_emergency_report
from events import Event, EventBus
from sensor_state import SensorFusionState
from system import CAMERA_ZONE, SENSOR_ZONE, AegisSystem


class FakeVision:
    def __init__(self, fire=0.0, smoke=0.0):
        self.fire, self.smoke = fire, smoke

    def latest_fire_smoke(self):
        return self.fire, self.smoke


def sensor_snap(level="NORMAL", score=0.0, anomaly=False, fire=0.0, smoke=0.0):
    return {
        "reading": {"temperature": 24.0, "smoke_level": 5.0, "gas_level": 10.0},
        "anomaly": anomaly, "risk_score": score, "risk_level": level,
        "trend": {"trend": "STABLE", "slope": 0.0, "predicted_next": score},
        "risk_history": [], "camera": {"fire": fire, "smoke": smoke},
    }


@pytest.fixture
def system():
    return AegisSystem(BuildingDigitalTwin(), SensorFusionState(), FakeVision())


def types(system):
    return [e["type"] for e in system.bus.timeline(limit=200)]


# --- event bus ---------------------------------------------------------------

def test_bus_orders_timeline_newest_first_and_assigns_ids():
    bus = EventBus()
    bus.publish(Event("A", "test", "first"))
    bus.publish(Event("B", "test", "second"))
    tl = bus.timeline()
    assert [e["type"] for e in tl] == ["B", "A"]
    assert tl[0]["id"] > tl[1]["id"]


def test_bus_isolates_failing_subscriber():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda e: 1 / 0)
    bus.subscribe(seen.append)
    bus.publish(Event("A", "test", "x"))
    assert len(seen) == 1


def test_bus_filters_by_type_and_rejects_bad_severity():
    bus = EventBus()
    seen = []
    bus.subscribe(seen.append, event_type="WANTED")
    bus.publish(Event("OTHER", "test", "x"))
    bus.publish(Event("WANTED", "test", "y"))
    assert [e.type for e in seen] == ["WANTED"]
    with pytest.raises(ValueError):
        bus.publish(Event("A", "test", "x", severity="catastrophic"))


def test_bus_timeline_is_bounded():
    bus = EventBus(history=5)
    for i in range(20):
        bus.publish(Event("A", "test", str(i)))
    assert len(bus.timeline(limit=100)) == 5


# --- operator actions --------------------------------------------------------

def test_operator_fire_updates_twin_and_timeline_once(system):
    assert system.start_fire("CorridorA")
    assert system.start_fire("CorridorA")  # idempotent
    assert system.twin.zone_status["CorridorA"] == "FIRE"
    assert types(system).count(ev.FIRE_STARTED) == 1


def test_unknown_zone_is_rejected(system):
    assert not system.start_fire("Atlantis")
    assert not system.clear_zone("Atlantis")
    assert types(system) == []


def test_clearing_a_safe_zone_logs_nothing(system):
    system.clear_zone("Room101")
    assert ev.ZONE_CLEARED not in types(system)


# --- reports -> proposed actions -> twin ----------------------------------------

def test_fire_report_proposes_but_does_not_change_twin(system):
    out = system.ingest_report(parse_emergency_report("Fire in Room 202, three people are trapped"))
    assert out["zones"] == ["Room202"]
    assert system.twin.zone_status["Room202"] == "SAFE"   # human must confirm first
    assert [p["zone"] for p in system.pending_actions()] == ["Room202"]
    assert ev.REPORT_PARSED in types(system) and ev.ACTION_PROPOSED in types(system)


def test_confirming_proposal_declares_fire(system):
    out = system.ingest_report(parse_emergency_report("Fire in Room 202"))
    pid = out["proposals"][0]["id"]
    system.decide(pid, approve=True)
    assert system.twin.zone_status["Room202"] == "FIRE"
    assert system.pending_actions() == []
    assert system.decide(pid, approve=True) is None       # can't decide twice


def test_dismissing_proposal_leaves_twin_untouched(system):
    out = system.ingest_report(parse_emergency_report("Fire in Room 202"))
    system.decide(out["proposals"][0]["id"], approve=False)
    assert system.twin.zone_status["Room202"] == "SAFE"
    assert ev.ACTION_DISMISSED in types(system)


def test_duplicate_reports_do_not_stack_proposals(system):
    system.ingest_report(parse_emergency_report("Fire in Room 202"))
    system.ingest_report(parse_emergency_report("Flames in room 202!"))
    assert len(system.pending_actions()) == 1


def test_smell_of_smoke_report_proposes_nothing(system):
    out = system.ingest_report(parse_emergency_report("Minor smoke smell in Room 101, not urgent"))
    assert out["proposals"] == []


def test_report_with_unmapped_location_is_still_recorded(system):
    out = system.ingest_report(parse_emergency_report("Fire in the cafeteria"))
    assert out["zones"] == [] and out["proposals"] == []
    assert len(system.reports()) == 1


# --- sensors -> twin ---------------------------------------------------------

def test_sensor_risk_is_mirrored_onto_zone_without_blocking_routes(system):
    system.on_sensor_update(sensor_snap("HIGH", 55.0))
    assert system.twin.zone_risk[SENSOR_ZONE] == 55.0
    assert system.twin.zone_status[SENSOR_ZONE] == "SAFE"
    assert system.twin.get_blocked_edges() == []


def test_sensor_level_changes_are_logged_once_per_transition(system):
    for _ in range(3):
        system.on_sensor_update(sensor_snap("HIGH", 55.0))
    system.on_sensor_update(sensor_snap("NORMAL", 3.0))
    msgs = [e["message"] for e in system.bus.timeline() if e["type"] == ev.SENSOR_RISK]
    assert len(msgs) == 2
    assert "fell to NORMAL" in msgs[0] and "rose to HIGH" in msgs[1]


def test_critical_sensors_propose_fire(system):
    system.on_sensor_update(sensor_snap("CRITICAL", 80.0))
    assert [p["zone"] for p in system.pending_actions()] == [SENSOR_ZONE]
    assert system.twin.zone_status[SENSOR_ZONE] == "SAFE"


# --- camera -> fusion / timeline -----------------------------------------------

def test_camera_detection_raises_event_and_proposal_with_hysteresis(system):
    system.on_sensor_update(sensor_snap(fire=0.8))
    system.on_sensor_update(sensor_snap(fire=0.5))   # between OFF and ON: still active, no new event
    system.on_sensor_update(sensor_snap(fire=0.8))
    assert types(system).count(ev.CAMERA_FIRE) == 1
    assert [p["zone"] for p in system.pending_actions()] == [CAMERA_ZONE]
    system.on_sensor_update(sensor_snap(fire=0.1))
    assert ev.CAMERA_CLEAR in types(system)


def test_camera_confidence_feeds_fused_risk_score():
    sensors = SensorFusionState()
    sensors.update_once(camera_fire_conf=0.0)
    without = sensors.get_snapshot()["risk_score"]
    sensors.update_once(camera_fire_conf=0.9)
    snap = sensors.get_snapshot()
    assert snap["camera"]["fire"] == 0.9
    assert snap["risk_score"] >= without + 30        # camera contributes up to 40 points


def test_no_vision_means_zero_camera_confidence():
    s = AegisSystem(BuildingDigitalTwin(), SensorFusionState(), vision=None)
    assert s.camera_confidence() == (0.0, 0.0)


# --- live agents ---------------------------------------------------------------

def test_agents_report_nothing_when_nothing_is_happening(system):
    assert "No declared fires" in FireAgent(system).report()
    assert "No reports of trapped" in MedicalAgent(system).report()
    assert "all standard evacuation routes are open" in RouteAgent(system).report()


def test_agents_reflect_live_twin_and_reports(system):
    system.start_fire("CorridorA")
    system.ingest_report(parse_emergency_report("Three people are trapped in Room 101"))

    fire = FireAgent(system).get_status()
    assert [f["zone"] for f in fire["declared_fires"]] == ["CorridorA"]

    med = MedicalAgent(system).get_status()
    assert med["people_reported_at_risk"][0]["people_count"] == 3
    assert med["people_reported_at_risk"][0]["zones"] == ["Room101"]
    assert med["people_reported_at_risk"][0]["verified"] is False
    assert med["medical_units_available"] is None     # unknown, never invented

    routes = RouteAgent(system).get_status()["routes"]
    # Room101 connects only to CorridorA, so with CorridorA burning it is isolated.
    assert routes["Room101"]["status"].startswith("ISOLATED")
    assert routes["Room103"]["status"] == "CLEAR"
