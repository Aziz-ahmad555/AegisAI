import random
from datetime import datetime, timedelta

def get_system_status():
    """Simulates the current overall system state, as if pulled live from Phases 4-6."""
    return {
        "timestamp": datetime.now().isoformat(),
        "overall_risk_score": 62.5,
        "overall_risk_level": "HIGH",
        "trend": "RISING",
    }


def get_active_incidents():
    """Simulates currently active incidents detected across the system."""
    return [
        {
            "id": "INC-001",
            "type": "FIRE",
            "location": "Building 4, Corridor A",
            "confidence": 0.78,
            "detected_at": (datetime.now() - timedelta(minutes=3)).isoformat(),
            "status": "ACTIVE",
        },
        {
            "id": "INC-002",
            "type": "TRAPPED_PERSON",
            "location": "Building 4, Gate 4",
            "confidence": 0.65,
            "detected_at": (datetime.now() - timedelta(minutes=2)).isoformat(),
            "status": "ACTIVE",
        },
    ]


def get_recent_reports():
    """Simulates recent NLP-parsed emergency reports from Phase 8."""
    return [
        {
            "text": "There is smoke coming from the second floor near the laboratory in Building 4. Three people are trapped near Gate 4.",
            "event_types": ["FIRE", "TRAPPED"],
            "severity": "CRITICAL",
            "people_count": 3,
            "locations": ["Building 4", "Gate 4", "laboratory"],
        }
    ]


def get_evacuation_status():
    """Simulates current evacuation routing state from Phase 6."""
    return {
        "affected_zone": "Corridor A",
        "primary_route_blocked": True,
        "recommended_route": "CorridorB -> ExitMain",
        "estimated_evacuation_time_minutes": 4.5,
    }
