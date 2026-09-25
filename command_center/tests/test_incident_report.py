"""Markdown incident report: built by code from the live system, escapes
everything that came from outside, labels simulated data, and downloads
only for a signed-in operator."""
import time

import pytest

from aegis_core import coordinator
from aegis_core.building_state import BuildingDigitalTwin
from aegis_core.emergency_nlp import parse_emergency_report
from aegis_core.incident_report import build_incident_report, md_escape, md_escape_rich
from aegis_core.scenario import FINISHED, ScenarioRunner
from aegis_core.sensor_state import SensorFusionState
from aegis_core.system import AegisSystem

from helpers import login

INJECTION = ("Fire in Room 101 | 3 people trapped <script>alert(1)</script> "
             "![x](http://evil.example/p.png) [click](http://evil.example)")


@pytest.fixture
def system():
    s = AegisSystem(BuildingDigitalTwin(), SensorFusionState())
    coordinator.init_agents(s)
    return s


def section(md, title):
    return md.split(f"## {title}", 1)[1].split("\n## ", 1)[0]


def test_empty_system_report_is_complete_and_says_so(system):
    md = build_incident_report(system)
    for heading in ("# Emergency Nexus incident report", "## Summary", "## Decisions", "## Caller reports",
                    "## Evacuation routes (at time of export)", "## Timeline"):
        assert heading in md
    assert "- **Declared fires:** none" in md
    assert "No actions were proposed." in md and "No caller reports." in md
    assert "SIMULATED" not in md
    assert "AI-generated" not in md                         # no model output without a briefing


def test_operator_incident_lists_fire_decisions_reports_and_routes(system):
    system.ingest_report(parse_emergency_report("Fire in Room 101, two people are trapped"))
    pid = system.pending_actions()[0]["id"]
    system.decide(pid, True)
    md = build_incident_report(system)

    assert "- **Declared fires:** Room101 (since " in md
    decisions = section(md, "Decisions")
    assert "| Proposed | reports | Room101 |" in decisions
    assert "| Confirmed | operator | Room101 | Operator confirmed: Caller reports fire in Room101" in decisions
    reports = section(md, "Caller reports")
    assert "Room101** - FIRE, TRAPPED, severity CRITICAL, people: 2, unverified" in reports
    routes = section(md, "Evacuation routes (at time of export)")
    assert "| Room101 | Through its own hazard zone | Room101 -\\> CorridorA -\\> ExitEmergency | ExitEmergency |" in routes
    timeline = section(md, "Timeline")
    # Oldest first: the report comes before the fire it led to.
    assert timeline.index("Report: FIRE, TRAPPED at Room101") < timeline.index("Fire declared in Room101")


def test_pending_proposals_are_flagged(system):
    system.ingest_report(parse_emergency_report("Fire in Room 202"))
    md = build_incident_report(system)
    assert "- **Actions awaiting confirmation:** 1" in md
    assert "**Still awaiting confirmation**" in section(md, "Decisions")


def test_caller_text_is_escaped_not_rendered(system):
    system.ingest_report(parse_emergency_report(INJECTION))
    md = build_incident_report(system)
    assert "<script>" not in md and "![x](" not in md and "[click](" not in md
    quote = next(line for line in section(md, "Caller reports").splitlines() if line.startswith("  > "))
    assert "&lt;script\\>" in quote and "\\!\\[x\\]" in quote and "\\|" in quote
    # Pipes can't break the timeline table either: every row keeps its 6 columns.
    for row in section(md, "Timeline").splitlines():
        if row.startswith("| "):
            assert row.replace("\\|", "").count("|") == 7


def test_md_escape_helpers():
    assert md_escape("a|b\nc") == "a\\|b c"
    assert md_escape("**x** [l](u)") == "\\*\\*x\\*\\* \\[l\\]\\(u\\)"
    rich = md_escape_rich("**Fire** [l](http://x) ![i](u) <img>")
    assert rich.startswith("**Fire**")                      # formatting kept
    assert "\\[l\\]" in rich and "\\!\\[i\\]" in rich and "<img>" not in rich


def test_only_the_incident_since_the_last_reset_is_reported(system):
    system.start_fire("Room202")
    system.reset_to_normal(source="operator")
    system.start_fire("Room103")
    timeline = section(build_incident_report(system), "Timeline")
    assert "Room103" in timeline and "Room202" not in timeline


def test_scenario_report_is_labelled_simulated_and_marks_the_briefing(system):
    runner = ScenarioRunner(system, time_scale=0)
    runner.start("kitchen_fire")
    end = time.time() + 15
    while runner.status != FINISHED and time.time() < end:
        time.sleep(0.02)
    md = build_incident_report(system, runner.state())
    assert '> **SIMULATED.** This report contains scripted data from the guided scenario "Kitchen fire".' in md
    briefing = section(md, "Agents' briefing (AI-generated)")
    assert "this section is model output" in briefing
    assert "> \\[OFFLINE MODE" in briefing                  # the offline briefing, quoted and escaped
    assert "Room102" in section(md, "Summary")


def test_download_requires_login(client):
    r = client.get("/api/incident-report.md")
    assert r.status_code in (302, 401)


def test_download_is_a_markdown_attachment(client):
    login(client)
    r = client.get("/api/incident-report.md")
    assert r.status_code == 200
    assert r.mimetype == "text/markdown"
    assert r.headers["Content-Disposition"].startswith('attachment; filename="emergency-nexus-incident-')
    assert r.headers["Cache-Control"] == "no-store"
    assert r.get_data(as_text=True).startswith("# Emergency Nexus incident report")


def test_pages_offer_the_export(client):
    login(client)
    assert b'href="/api/incident-report.md"' in client.get("/").data
    assert b'id="sc-export"' in client.get("/chat").data
