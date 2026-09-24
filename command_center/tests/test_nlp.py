"""Emergency report parsing: event types, severity (incl. downplaying), people, locations."""
import pytest

from aegis_core.emergency_nlp import parse_emergency_report as parse


@pytest.mark.parametrize("text, events, severity", [
    ("There is smoke coming from the second floor near the laboratory. Three people are trapped.",
     ["FIRE", "TRAPPED"], "CRITICAL"),
    ("I see a small fire in the kitchen, nothing major, just want someone to check it out.",
     ["FIRE"], "MODERATE"),                                     # downplayed HIGH -> MODERATE
    ("Minor smoke smell in the break room, not urgent.", ["POSSIBLE_FIRE"], "MODERATE"),
    ("Someone collapsed and is unconscious near the cafeteria entrance.", ["MEDICAL"], "CRITICAL"),
    ("The building has collapsed and people are under the debris.", ["STRUCTURAL"], "CRITICAL"),
    ("12 people stuck in the elevator", ["TRAPPED"], "CRITICAL"),   # regression: was LOW
    ("The water is rising in the basement", ["FLOOD"], "LOW"),     # regression: was UNKNOWN
    ("hello there", ["UNKNOWN"], "LOW"),
])
def test_event_type_and_severity(text, events, severity):
    r = parse(text)
    assert r["event_types"] == events
    assert r["severity"] == severity


def test_confirmed_fire_wins_over_possible_fire():
    assert parse("Smell of smoke and now visible flames in the lobby")["event_types"] == ["FIRE"]


def test_downplaying_never_hides_critical_as_low():
    # Downplay softens CRITICAL/HIGH to MODERATE, never below.
    assert parse("Someone is trapped but it's not serious")["severity"] == "MODERATE"


@pytest.mark.parametrize("text, count", [
    ("Three people are trapped", 3),
    ("12 people stuck", 12),
    ("ten victims near the exit", 10),
    ("two individuals injured", 2),
    ("people are trapped", None),
    ("3 cars on fire", None),
])
def test_people_count(text, count):
    assert parse(text)["people_count"] == count


def test_longest_room_word_wins_without_double_matching():
    locs = parse("Fire in the laboratory")["locations"]
    assert "laboratory" in locs and "lab" not in locs


def test_multiple_room_words_are_all_found():
    locs = parse("Smoke in the hallway spreading to the kitchen")["locations"]
    assert {"hallway", "kitchen"} <= set(locs)


def test_original_text_is_preserved():
    text = "Fire <b>in</b> the lobby"
    assert parse(text)["original_text"] == text
