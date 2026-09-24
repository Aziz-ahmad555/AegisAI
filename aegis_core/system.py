import itertools
import os
import re
import threading
import time

from . import events as ev
from .events import Event, EventBus

# Which twin zone the (simulated) sensor cluster and the camera watch.
# Real deployments would map many sensors/cameras to many zones; this demo
# has one of each, so each gets a single configurable zone.
SENSOR_ZONE = os.environ.get("AEGISAI_SENSOR_ZONE", "CorridorB")
CAMERA_ZONE = os.environ.get("AEGISAI_CAMERA_ZONE", "Room201")

# Camera fire/smoke confidence needed to raise / clear a CAMERA_FIRE event.
# Two thresholds (hysteresis) so a detection flickering around one value
# doesn't spam the timeline.
CAMERA_ON = 0.55
CAMERA_OFF = 0.35
RISK_ORDER = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]


class AegisSystem:
    """
    Connects the independent modules into one system:

      vision  --fire/smoke conf-->  sensor fusion  --risk level-->  twin zone risk
      reports --parsed location-->  proposed action --operator-->   twin fire state
      twin / sensors / vision / reports  --events-->  timeline + live agents

    Anything that would change the building's emergency state on the basis
    of automated evidence (a sensor spike, a camera detection, a caller's
    report) is only *proposed*; an operator confirms it. Operators acting
    directly on the twin are applied immediately.
    """

    def __init__(self, twin, sensors, vision=None, bus=None):
        self.twin = twin
        self.sensors = sensors
        self.vision = vision
        self.bus = bus or EventBus()
        self._lock = threading.RLock()
        self._action_ids = itertools.count(1)
        self._pending = {}              # id -> proposed action dict
        self._reports = []              # parsed reports, newest last
        self._camera_active = False
        self._sensor_level = "NORMAL"
        self._sensor_anomaly = False

    # ----- zones -----------------------------------------------------------

    @staticmethod
    def sensor_zone():
        return SENSOR_ZONE

    @staticmethod
    def camera_zone():
        return CAMERA_ZONE

    def zones(self):
        return list(self.twin.graph.nodes())

    def match_zone(self, text):
        """Map free text like 'room 101' / 'corridor a' / 'stairwell' to a twin zone."""
        norm = re.sub(r"[^a-z0-9]", "", text.lower())
        for zone in self.zones():
            if zone.lower() == norm:
                return zone
        return None

    def zones_in_text(self, text):
        found = []
        squashed = re.sub(r"[^a-z0-9]", "", text.lower())
        for zone in self.zones():
            if zone.lower() in squashed and zone not in found:
                found.append(zone)
        return found

    # ----- operator actions on the twin -------------------------------------

    def start_fire(self, zone, source="operator", reason=None):
        if zone not in self.zones():
            return False
        if self.twin.zone_status.get(zone) == "FIRE":
            return True
        self.twin.start_fire(zone)
        self.bus.publish(Event(ev.FIRE_STARTED, source, reason or f"Fire declared in {zone}",
                               severity="critical", zone=zone))
        return True

    def clear_zone(self, zone, source="operator"):
        if zone not in self.zones():
            return False
        was_fire = self.twin.zone_status.get(zone) == "FIRE"
        self.twin.clear_zone(zone)
        if was_fire:
            self.bus.publish(Event(ev.ZONE_CLEARED, source, f"{zone} confirmed clear", zone=zone))
        return True

    # ----- proposed actions (need operator confirmation) --------------------

    def propose(self, action, zone, reason, source, confidence=None, evidence=None):
        with self._lock:
            for p in self._pending.values():
                if p["action"] == action and p["zone"] == zone:
                    return p          # already awaiting a decision
            pid = next(self._action_ids)
            proposal = {
                "id": pid, "action": action, "zone": zone, "reason": reason,
                "source": source, "confidence": confidence, "evidence": evidence or {},
                "ts": time.time(),
            }
            self._pending[pid] = proposal
        self.bus.publish(Event(ev.ACTION_PROPOSED, source, reason, severity="warning", zone=zone,
                               confidence=confidence, data={"proposal_id": pid, "action": action}))
        return proposal

    def pending_actions(self):
        with self._lock:
            return sorted(self._pending.values(), key=lambda p: p["ts"])

    def decide(self, proposal_id, approve):
        with self._lock:
            proposal = self._pending.pop(proposal_id, None)
        if proposal is None:
            return None
        zone = proposal["zone"]
        if approve:
            self.bus.publish(Event(ev.ACTION_CONFIRMED, "operator",
                                   f"Operator confirmed: {proposal['reason']}", zone=zone,
                                   data={"proposal_id": proposal_id}))
            if proposal["action"] == "declare_fire":
                self.start_fire(zone, source="operator",
                                reason=f"Fire declared in {zone} (confirmed from {proposal['source']})")
        else:
            self.bus.publish(Event(ev.ACTION_DISMISSED, "operator",
                                   f"Operator dismissed: {proposal['reason']}", zone=zone,
                                   data={"proposal_id": proposal_id}))
        return proposal

    # ----- camera -> fusion --------------------------------------------------

    def camera_confidence(self):
        """(fire, smoke) confidence from Live Vision, 0 when no fresh detection."""
        if self.vision is None:
            return 0.0, 0.0
        return self.vision.latest_fire_smoke()

    def _check_camera(self, fire, smoke):
        conf = max(fire, smoke)
        if not self._camera_active and conf >= CAMERA_ON:
            self._camera_active = True
            kind = "Fire" if fire >= smoke else "Smoke"
            self.bus.publish(Event(ev.CAMERA_FIRE, "vision",
                                   f"{kind} detected on camera watching {CAMERA_ZONE}",
                                   severity="warning", zone=CAMERA_ZONE, confidence=round(conf, 2)))
            self.propose("declare_fire", CAMERA_ZONE,
                         f"Camera detected {kind.lower()} in {CAMERA_ZONE} - declare fire?",
                         source="vision", confidence=round(conf, 2))
        elif self._camera_active and conf < CAMERA_OFF:
            self._camera_active = False
            self.bus.publish(Event(ev.CAMERA_CLEAR, "vision",
                                   f"Camera no longer sees fire/smoke in {CAMERA_ZONE}", zone=CAMERA_ZONE))

    # ----- sensors -> twin ---------------------------------------------------

    def on_sensor_update(self, snap):
        """Called by the sensor loop after every fused reading."""
        fire, smoke = snap.get("camera", {}).get("fire", 0.0), snap.get("camera", {}).get("smoke", 0.0)
        self._check_camera(fire, smoke)

        level = snap["risk_level"]
        # Mirror the fused sensor risk onto the monitored zone, unless that
        # zone is already a declared fire (which carries its own risk).
        if self.twin.zone_status.get(SENSOR_ZONE) != "FIRE":
            self.twin.set_zone_risk(SENSOR_ZONE, snap["risk_score"])

        if level != self._sensor_level:
            rising = RISK_ORDER.index(level) > RISK_ORDER.index(self._sensor_level)
            self._sensor_level = level
            severity = {"NORMAL": "info", "ELEVATED": "info", "HIGH": "warning", "CRITICAL": "critical"}[level]
            verb = "rose" if rising else "fell"
            self.bus.publish(Event(ev.SENSOR_RISK, "sensors",
                                   f"Sensor risk in {SENSOR_ZONE} {verb} to {level} ({snap['risk_score']})",
                                   severity=severity, zone=SENSOR_ZONE, confidence=None,
                                   data={"risk_score": snap["risk_score"], "reading": snap["reading"]}))
            if level == "CRITICAL" and self.twin.zone_status.get(SENSOR_ZONE) != "FIRE":
                self.propose("declare_fire", SENSOR_ZONE,
                             f"Sensors in {SENSOR_ZONE} reached CRITICAL ({snap['risk_score']}) - declare fire?",
                             source="sensors", evidence={"reading": snap["reading"]})

        if snap["anomaly"] and not self._sensor_anomaly:
            self.bus.publish(Event(ev.SENSOR_ANOMALY, "sensors",
                                   f"Anomalous sensor reading in {SENSOR_ZONE}",
                                   severity="warning", zone=SENSOR_ZONE, data={"reading": snap["reading"]}))
        self._sensor_anomaly = snap["anomaly"]

    # ----- reports -> twin ---------------------------------------------------

    def ingest_report(self, parsed):
        """Record a parsed emergency report and propose actions it implies."""
        zones = []
        for loc in parsed["locations"]:
            z = self.match_zone(loc)
            if z and z not in zones:
                zones.append(z)
        for z in self.zones_in_text(parsed["original_text"]):
            if z not in zones:
                zones.append(z)

        record = {
            "text": parsed["original_text"], "event_types": parsed["event_types"],
            "severity": parsed["severity"], "people_count": parsed["people_count"],
            "zones": zones, "ts": time.time(), "verified": False,
        }
        with self._lock:
            self._reports.append(record)
            self._reports = self._reports[-50:]

        severity = {"CRITICAL": "critical", "HIGH": "warning"}.get(parsed["severity"], "info")
        where = ", ".join(zones) if zones else "unmapped location"
        self.bus.publish(Event(ev.REPORT_PARSED, "reports",
                               f"Report: {', '.join(parsed['event_types'])} at {where} ({parsed['severity']})",
                               severity=severity, zone=zones[0] if zones else None,
                               data={"people_count": parsed["people_count"], "zones": zones}))

        proposals = []
        if "FIRE" in parsed["event_types"]:
            for z in zones:
                if self.twin.zone_status.get(z) != "FIRE":
                    proposals.append(self.propose("declare_fire", z,
                                                  f"Caller reports fire in {z} - declare fire?",
                                                  source="reports", evidence={"report": parsed["original_text"]}))
        return {"zones": zones, "proposals": proposals}

    def reports(self):
        with self._lock:
            return list(self._reports)

    # ----- snapshot ----------------------------------------------------------

    def status(self):
        fire_zones = [z for z, s in self.twin.zone_status.items() if s == "FIRE"]
        snap = self.sensors.get_snapshot()
        return {
            "fire_zones": fire_zones,
            "sensor_zone": SENSOR_ZONE,
            "camera_zone": CAMERA_ZONE,
            "sensor_level": snap["risk_level"],
            "sensor_score": snap["risk_score"],
            "camera_active": self._camera_active,
            "pending_actions": self.pending_actions(),
        }
