"""Digital twin evacuation routing: the three documented cases plus state handling."""
import threading

import pytest

from aegis_core.building_state import BuildingDigitalTwin


@pytest.fixture
def twin():
    return BuildingDigitalTwin()


def test_clean_route_when_nothing_is_burning(twin):
    r = twin.compute_evacuation_route("Room101")
    assert r["path"] == ["Room101", "CorridorA", "ExitEmergency"]
    assert r["exit"] == "ExitEmergency" and r["through_hazard"] is False
    assert r["length"] == 4


def test_reroutes_around_an_unrelated_fire(twin):
    twin.start_fire("CorridorB")
    # Room201 connects only to CorridorB, so it is isolated...
    assert twin.compute_evacuation_route("Room201")["path"] is None
    # ...while Room202 has another way out and must not use CorridorB.
    r = twin.compute_evacuation_route("Room202")
    assert "CorridorB" not in r["path"] and r["through_hazard"] is False


def test_route_through_own_hazard_zone_is_flagged(twin):
    twin.start_fire("Room103")
    r = twin.compute_evacuation_route("Room103")
    assert r["path"] == ["Room103", "ExitEmergency"]
    assert r["through_hazard"] is True


def test_full_isolation_is_reported_not_hidden(twin):
    twin.start_fire("CorridorA")
    r = twin.compute_evacuation_route("Room101")
    assert r == {"path": None, "length": None, "exit": None, "through_hazard": False}


def test_other_fire_zones_are_never_routed_through(twin):
    twin.start_fire("Room202")
    twin.start_fire("CorridorB")
    r = twin.compute_evacuation_route("Room202")
    # Room202 may leave via its own door, but never through CorridorB.
    assert r["path"] is not None and "CorridorB" not in r["path"]
    assert r["through_hazard"] is True


def test_clearing_restores_the_original_route(twin):
    before = twin.compute_evacuation_route("Room101")
    twin.start_fire("CorridorA")
    twin.clear_zone("CorridorA")
    assert twin.compute_evacuation_route("Room101") == before
    assert twin.get_blocked_edges() == []


def test_blocked_edges_are_exactly_those_touching_fire(twin):
    twin.start_fire("Stairwell")
    blocked = {frozenset(e) for e in twin.get_blocked_edges()}
    assert blocked == {frozenset(e) for e in [("CorridorB", "Stairwell"), ("Stairwell", "Room202"),
                                              ("Stairwell", "ExitEmergency")]}


def test_unknown_zone_is_rejected(twin):
    assert twin.start_fire("Atlantis") is False
    assert twin.clear_zone("Atlantis") is False


def test_monitored_risk_does_not_override_a_declared_fire(twin):
    twin.start_fire("CorridorA")
    twin.set_zone_risk("CorridorA", 10)
    assert twin.zone_risk["CorridorA"] == 85.0
    twin.set_zone_risk("CorridorB", 42.36)
    assert twin.zone_risk["CorridorB"] == 42.4


def test_overall_risk_is_mean_zone_risk(twin):
    twin.start_fire("CorridorA")
    assert twin.get_overall_risk() == round(85.0 / len(twin.zone_risk), 1)


def test_event_log_is_bounded(twin):
    for _ in range(30):
        twin.start_fire("Room101")
        twin.clear_zone("Room101")
    assert len(twin.event_log) == 20
    assert len(twin.get_state_snapshot()["event_log"]) == 10


def test_concurrent_mutation_and_reads_do_not_deadlock_or_crash(twin):
    # Regression guard for the Phase 10 deadlock (non-reentrant lock taken
    # twice by get_state_snapshot -> get_blocked_edges).
    errors = []
    zones = ["Room101", "CorridorA", "CorridorB", "Stairwell", "Room202"]

    def writer(i):
        try:
            for n in range(200):
                z = zones[(i + n) % len(zones)]
                (twin.start_fire if n % 2 == 0 else twin.clear_zone)(z)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                twin.get_state_snapshot()
                twin.compute_evacuation_route("Room101")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads), "deadlock: threads still running"
    assert errors == []
