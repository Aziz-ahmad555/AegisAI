import time


def _ago(ts):
    minutes = (time.time() - ts) / 60
    return round(minutes, 1)


class FireAgent:
    """Fire/smoke picture: declared fires on the twin, live camera detection,
    and fused sensor risk for the monitored zone."""

    def __init__(self, system):
        self.system = system

    def get_status(self):
        s = self.system
        twin = s.twin
        fires = [
            {"zone": z, "declared_minutes_ago": _ago(twin.fire_since[z]) if z in twin.fire_since else None}
            for z, status in twin.zone_status.items() if status == "FIRE"
        ]
        snap = s.sensors.get_snapshot()
        fire_conf, smoke_conf = s.camera_confidence()
        camera_event = s.bus.latest("CAMERA_FIRE")
        return {
            "agent": "FireAgent",
            "declared_fires": fires,
            "camera": {
                "zone": s.camera_zone(),
                "fire_confidence": round(fire_conf, 2),
                "smoke_confidence": round(smoke_conf, 2),
                "detecting_now": s.status()["camera_active"],
                "last_detection_minutes_ago": _ago(camera_event.ts) if camera_event else None,
                "note": "camera confidence is only available while Live Vision runs in fire/smoke mode",
            },
            "sensors": {
                "zone": s.sensor_zone(),
                "risk_score": snap["risk_score"],
                "risk_level": snap["risk_level"],
                "trend": snap["trend"]["trend"],
                "anomaly": snap["anomaly"],
                "reading": snap["reading"],
            },
        }

    def report(self):
        st = self.get_status()
        parts = []
        if st["declared_fires"]:
            zones = ", ".join(
                f"{f['zone']} ({f['declared_minutes_ago']} min ago)" if f["declared_minutes_ago"] is not None else f["zone"]
                for f in st["declared_fires"]
            )
            parts.append(f"{len(st['declared_fires'])} declared fire(s): {zones}.")
        else:
            parts.append("No declared fires.")
        sen = st["sensors"]
        parts.append(f"Sensors in {sen['zone']}: {sen['risk_level']} (score {sen['risk_score']}, trend {sen['trend']}"
                     f"{', anomaly' if sen['anomaly'] else ''}).")
        cam = st["camera"]
        if cam["detecting_now"]:
            parts.append(f"Camera on {cam['zone']} currently detects fire {cam['fire_confidence']} / smoke {cam['smoke_confidence']}.")
        else:
            parts.append(f"Camera on {cam['zone']}: no current fire/smoke detection.")
        return "FireAgent: " + " ".join(parts)


class MedicalAgent:
    """People at risk, from caller reports. Report data is unverified by
    definition and is always labelled as such."""

    def __init__(self, system):
        self.system = system

    def get_status(self):
        reports = self.system.reports()
        people = []
        for r in reports:
            relevant = set(r["event_types"]) & {"TRAPPED", "MEDICAL", "STRUCTURAL"}
            if relevant or r["people_count"]:
                people.append({
                    "zones": r["zones"] or ["unmapped location"],
                    "people_count": r["people_count"],
                    "event_types": r["event_types"],
                    "severity": r["severity"],
                    "reported_minutes_ago": _ago(r["ts"]),
                    "verified": r["verified"],
                    "report": r["text"],
                })
        fire_zones = set(self.system.status()["fire_zones"])
        return {
            "agent": "MedicalAgent",
            "reports_on_file": len(reports),
            "people_reported_at_risk": people,
            "people_in_fire_zones": [p for p in people if fire_zones & set(p["zones"])],
            "medical_units_available": None,
            "note": "people counts come from unverified caller reports; medical unit availability is not tracked by this system",
        }

    def report(self):
        st = self.get_status()
        if not st["people_reported_at_risk"]:
            return f"MedicalAgent: No reports of trapped or injured people ({st['reports_on_file']} report(s) on file)."
        lines = []
        for p in st["people_reported_at_risk"]:
            count = f"{p['people_count']} people" if p["people_count"] else "unknown number of people"
            lines.append(f"{count} - {', '.join(p['event_types'])} at {', '.join(p['zones'])} "
                         f"(reported {p['reported_minutes_ago']} min ago, unverified)")
        return "MedicalAgent: " + "; ".join(lines) + ". Medical unit availability: unknown."


class RouteAgent:
    """Evacuation routes from every occupiable room given the current fire state."""

    def __init__(self, system):
        self.system = system

    def get_status(self):
        twin = self.system.twin
        rooms = [z for z in twin.graph.nodes() if z.startswith("Room")]
        routes = {}
        for room in rooms:
            r = twin.compute_evacuation_route(room)
            if r["path"] is None:
                routes[room] = {"status": "ISOLATED - rescue required"}
            else:
                routes[room] = {
                    "status": "THROUGH OWN HAZARD ZONE" if r["through_hazard"] else "CLEAR",
                    "path": " -> ".join(r["path"]),
                    "exit": r["exit"],
                    "length": r["length"],
                }
        return {
            "agent": "RouteAgent",
            "blocked_zones": [z for z, s in twin.zone_status.items() if s == "FIRE"],
            "blocked_connections": [f"{u}-{v}" for u, v in twin.get_blocked_edges()],
            "routes": routes,
        }

    def report(self):
        st = self.get_status()
        if not st["blocked_zones"]:
            return "RouteAgent: No zones blocked - all standard evacuation routes are open."
        parts = [f"Blocked zones: {', '.join(st['blocked_zones'])}."]
        for room, r in st["routes"].items():
            if r["status"] == "CLEAR":
                parts.append(f"{room}: {r['path']}.")
            else:
                parts.append(f"{room}: {r['status']}{' via ' + r['path'] if 'path' in r else ''}.")
        return "RouteAgent: " + " ".join(parts)
