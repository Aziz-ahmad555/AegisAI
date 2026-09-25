"""
Markdown incident report built from the live system: current situation,
operator decisions, caller reports, evacuation routes and the full timeline.

Deterministic - no LLM writes any part of it. The only model text it can
contain is the scenario briefing, included verbatim under a heading that says
so. Everything that came from outside (caller text, model text, event
messages) is escaped, so it reads as text in any Markdown viewer and can't
inject links, images, HTML or table breaks.
"""
import re
import time

from . import events as ev

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|>~])")
_MD_LINKISH = re.compile(r"([\\\[\]!])")


def md_escape(text):
    """Escape a string for use as plain text inside Markdown (incl. table cells)."""
    text = " ".join(str(text).split())                  # no line breaks inside a cell/line
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    return _MD_SPECIAL.sub(r"\\\1", text)


def md_escape_rich(text):
    """For model-written Markdown: keeps emphasis and lists, but neutralises
    HTML, links and images (&, <, [, ], ! escaped)."""
    text = str(text).replace("&", "&amp;").replace("<", "&lt;")
    return _MD_LINKISH.sub(lambda m: chr(92) + m.group(1), text)


def _stamp(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _clock(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _row(*cells):
    return "| " + " | ".join(cells) + " |"


def build_incident_report(system, scenario_state=None, now=None, limit=200):
    """The report as a Markdown string. scenario_state is ScenarioRunner.state()."""
    now = now or time.time()
    events = list(reversed(system.bus.timeline(limit=limit)))         # oldest first
    # Only the incident since the last return-to-normal belongs in the report.
    last_reset = max((i for i, e in enumerate(events) if e["type"] == ev.RESET), default=-1)
    events = events[last_reset + 1:]
    simulated = any(e["data"].get("simulated") for e in events)
    status = system.status()
    reports = system.reports()
    pending = system.pending_actions()
    scenario_active = bool(scenario_state and scenario_state.get("active"))

    out = ["# AegisAI incident report", ""]
    out.append(f"Generated {_stamp(now)} (local time) by the AegisAI command center. "
               "Built from the system's own records; no part of this report is written by an AI model "
               "except where a section says so.")
    out.append("")
    if simulated or scenario_active:
        title = scenario_state.get("title") if scenario_active else None
        what = f"the guided scenario \"{md_escape(title)}\"" if title else "a guided scenario"
        out += [f"> **SIMULATED.** This report contains scripted data from {what}. It does not describe a real incident.", ""]

    # --- summary -------------------------------------------------------------------
    out += ["## Summary", ""]
    fires = status["fire_zones"]
    since = system.twin.fire_since
    if fires:
        out.append("- **Declared fires:** " + ", ".join(
            f"{md_escape(z)} (since {_clock(since[z])})" if z in since else md_escape(z) for z in fires))
    else:
        out.append("- **Declared fires:** none")
    snap = system.sensors.get_snapshot()
    out.append(f"- **Sensors ({md_escape(system.sensor_zone())}):** {md_escape(snap['risk_level'])}, "
               f"risk score {snap['risk_score']}, trend {md_escape(snap['trend']['trend'])}")
    out.append(f"- **Caller reports:** {len(reports)} (unverified)")
    out.append(f"- **Actions awaiting confirmation:** {len(pending)}")
    if events:
        out.append(f"- **Period covered:** {_stamp(events[0]['ts'])} to {_stamp(events[-1]['ts'])} "
                   f"({len(events)} events)")
    else:
        out.append("- **Period covered:** no events recorded since the last reset")
    out.append("")

    # --- decisions -------------------------------------------------------------------
    out += ["## Decisions", ""]
    decisions = [e for e in events if e["type"] in (ev.ACTION_PROPOSED, ev.ACTION_CONFIRMED, ev.ACTION_DISMISSED)]
    if decisions or pending:
        out += [_row("Time", "Decision", "By", "Zone", "Detail"), _row("---", "---", "---", "---", "---")]
        labels = {ev.ACTION_PROPOSED: "Proposed", ev.ACTION_CONFIRMED: "Confirmed", ev.ACTION_DISMISSED: "Dismissed"}
        for e in decisions:
            out.append(_row(e["time"], labels[e["type"]], md_escape(e["source"]), md_escape(e["zone"] or "-"),
                            md_escape(e["message"])))
        for p in pending:
            out.append(_row(_clock(p["ts"]), "**Still awaiting confirmation**", md_escape(p["source"]),
                            md_escape(p["zone"]), md_escape(p["reason"])))
    else:
        out.append("No actions were proposed.")
    out.append("")

    # --- caller reports --------------------------------------------------------------
    out += ["## Caller reports", "",
            "Parsed by the report analyzer. Reports are unverified, and the caller's words are quoted as "
            "received.", ""]
    if reports:
        for r in reports:
            zones = ", ".join(md_escape(z) for z in r["zones"]) or "unmapped location"
            people = r["people_count"] if r["people_count"] is not None else "not stated"
            out.append(f"- **{_clock(r['ts'])} · {zones}** - {md_escape(', '.join(r['event_types']) or 'unclassified')}, "
                       f"severity {md_escape(r['severity'])}, people: {people}, "
                       f"{'verified' if r['verified'] else 'unverified'}")
            out.append(f"  > {md_escape(r['text'])}")
    else:
        out.append("No caller reports.")
    out.append("")

    # --- routes ---------------------------------------------------------------------
    out += ["## Evacuation routes (at time of export)", "",
            _row("Room", "Status", "Route", "Exit"), _row("---", "---", "---", "---")]
    for room in [z for z in system.zones() if z.startswith("Room")]:
        r = system.twin.compute_evacuation_route(room)
        if r["path"] is None:
            out.append(_row(room, "**Isolated - rescue required**", "-", "-"))
        else:
            state = "Through its own hazard zone" if r["through_hazard"] else "Clear"
            out.append(_row(room, state, md_escape(" -> ".join(r["path"])), md_escape(r["exit"])))
    out.append("")

    # --- briefing ---------------------------------------------------------------------
    briefing = scenario_state.get("briefing") if scenario_active else None
    if briefing:
        out += ["## Agents' briefing (AI-generated)", "",
                "Written by the decision agent from live agent data during the scenario. "
                "Unlike the rest of this report, this section is model output.", ""]
        out += ["> " + md_escape_rich(line) if line.strip() else ">" for line in briefing.strip().splitlines()]
        out.append("")

    # --- timeline ---------------------------------------------------------------------
    out += ["## Timeline", ""]
    if events:
        out += [_row("Time", "Severity", "Source", "Zone", "Event", "Confidence"),
                _row("---", "---", "---", "---", "---", "---")]
        for e in events:
            conf = f"{e['confidence']:.2f}" if e["confidence"] is not None else "-"
            sev = e["severity"].upper() if e["severity"] != "info" else "info"
            out.append(_row(e["time"], sev, md_escape(e["source"]), md_escape(e["zone"] or "-"),
                            md_escape(e["message"]), conf))
    else:
        out.append("No events recorded since the last reset.")
    out += ["", "---", "", "*AegisAI is a portfolio demonstration, not a certified safety system.*", ""]
    return "\n".join(out)
