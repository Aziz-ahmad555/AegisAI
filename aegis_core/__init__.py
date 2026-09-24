"""
AegisAI core: the domain logic behind the Command Center.

    building_state   digital twin of the building + evacuation routing
    sensor_*         simulated IoT sensors, fused risk score (fusion_engine),
                     anomaly_detector, trend_predictor
    emergency_nlp    emergency-report parsing
    events / system  event bus, incident timeline, module wiring
    agents / coordinator   Fire / Medical / Route agents + LLM decision agent
    vision_stream    low-latency webcam detection pipeline (needs torch)

Modules are imported explicitly (e.g. ``from aegis_core.system import
AegisSystem``) so hosted deployments never pull in the vision stack.
"""
