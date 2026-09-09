import random
from datetime import datetime, timedelta


class FireAgent:
    """Specialist agent focused on fire/smoke detection and risk assessment."""

    def get_status(self):
        return {
            "agent": "FireAgent",
            "active_fires": [
                {"location": "Building 4, Corridor A", "confidence": 0.78, "detected_minutes_ago": 3}
            ],
            "fire_risk_score": 78.0,
            "trend": "RISING",
        }

    def report(self):
        status = self.get_status()
        if not status["active_fires"]:
            return "FireAgent: No active fires detected. Risk score nominal."
        fires = status["active_fires"]
        summary = f"FireAgent: {len(fires)} active fire(s) detected. "
        summary += f"Highest confidence: {fires[0]['location']} (confidence {fires[0]['confidence']}, detected {fires[0]['detected_minutes_ago']} min ago). "
        summary += f"Fire risk score: {status['fire_risk_score']}, trend: {status['trend']}."
        return summary


class MedicalAgent:
    """Specialist agent focused on injuries, trapped people, and medical needs."""

    def get_status(self):
        return {
            "agent": "MedicalAgent",
            "reported_injuries": 0,
            "trapped_people": 3,
            "trapped_location": "Building 4, Gate 4",
            "medical_units_available": 4,
        }

    def report(self):
        status = self.get_status()
        summary = f"MedicalAgent: {status['trapped_people']} people reported trapped at {status['trapped_location']}. "
        summary += f"{status['reported_injuries']} confirmed injuries so far. "
        summary += f"{status['medical_units_available']} medical units currently available for dispatch."
        return summary


class RouteAgent:
    """Specialist agent focused on evacuation routing status."""

    def get_status(self):
        return {
            "agent": "RouteAgent",
            "affected_zone": "Corridor A",
            "primary_route_blocked": True,
            "recommended_route": "CorridorB -> ExitMain",
            "estimated_evacuation_time_minutes": 4.5,
        }

    def report(self):
        status = self.get_status()
        if status["primary_route_blocked"]:
            summary = f"RouteAgent: Primary evacuation route through {status['affected_zone']} is BLOCKED. "
            summary += f"Recommended alternate route: {status['recommended_route']} (est. {status['estimated_evacuation_time_minutes']} min)."
        else:
            summary = "RouteAgent: All evacuation routes clear."
        return summary


if __name__ == "__main__":
    fire = FireAgent()
    medical = MedicalAgent()
    route = RouteAgent()

    print(fire.report())
    print(medical.report())
    print(route.report())
