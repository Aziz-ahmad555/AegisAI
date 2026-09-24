import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict


# Event types published by the modules. Kept as plain strings so they
# serialize straight into the timeline JSON.
FIRE_STARTED = "FIRE_STARTED"            # a twin zone was set on fire (operator or confirmed report)
ZONE_CLEARED = "ZONE_CLEARED"
CAMERA_FIRE = "CAMERA_FIRE"              # Live Vision's fire/smoke model detected fire/smoke
CAMERA_CLEAR = "CAMERA_CLEAR"
SENSOR_RISK = "SENSOR_RISK"              # fused sensor risk level changed
SENSOR_ANOMALY = "SENSOR_ANOMALY"
REPORT_PARSED = "REPORT_PARSED"          # an emergency report was analyzed
ACTION_PROPOSED = "ACTION_PROPOSED"      # the system suggests an action; needs operator confirmation
ACTION_CONFIRMED = "ACTION_CONFIRMED"
ACTION_DISMISSED = "ACTION_DISMISSED"

SEVERITIES = ("info", "warning", "critical")


@dataclass
class Event:
    type: str
    source: str                  # "vision" | "sensors" | "twin" | "reports" | "operator"
    message: str
    severity: str = "info"
    zone: str | None = None
    confidence: float | None = None
    data: dict = field(default_factory=dict)
    id: int = 0
    ts: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["time"] = time.strftime("%H:%M:%S", time.localtime(self.ts))
        return d


class EventBus:
    """
    In-process publish/subscribe hub plus the unified incident timeline.

    Every module publishes what it observed or did; subscribers (other
    modules, the WebSocket broadcaster) react. Handlers run synchronously on
    the publisher's thread, so they must be quick and must not publish
    recursively in a loop. A failing handler is isolated - it never breaks
    the publisher or other subscribers.
    """

    def __init__(self, history=200):
        self._lock = threading.RLock()
        self._subscribers = []   # (event_type or None for all, handler)
        self._timeline = deque(maxlen=history)
        self._ids = itertools.count(1)

    def subscribe(self, handler, event_type=None):
        with self._lock:
            self._subscribers.append((event_type, handler))

    def publish(self, event):
        if event.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {event.severity!r}")
        with self._lock:
            event.id = next(self._ids)
            event.ts = event.ts or time.time()
            self._timeline.append(event)
            handlers = [h for t, h in self._subscribers if t is None or t == event.type]
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:  # a broken subscriber must not take down the publisher
                print(f"[events] handler {getattr(handler, '__name__', handler)} failed on {event.type}: {e}")
        return event

    def timeline(self, limit=50):
        with self._lock:
            items = list(self._timeline)[-limit:]
        return [e.to_dict() for e in reversed(items)]

    def latest(self, event_type, zone=None):
        with self._lock:
            for e in reversed(self._timeline):
                if e.type == event_type and (zone is None or e.zone == zone):
                    return e
        return None
