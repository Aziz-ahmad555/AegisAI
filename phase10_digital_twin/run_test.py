import json
from building_state import BuildingDigitalTwin

twin = BuildingDigitalTwin()
print("Initial state:")
print(json.dumps(twin.get_state_snapshot(), indent=2), flush=True)

print("\nStarting fire in CorridorA...", flush=True)
twin.start_fire("CorridorA")

print("\nComputing evacuation route from Room101...", flush=True)
route = twin.compute_evacuation_route("Room101")
print(f"Route: {route}", flush=True)

print(f"\nOverall risk: {twin.get_overall_risk()}", flush=True)

print("\nClearing CorridorA...", flush=True)
twin.clear_zone("CorridorA")
route2 = twin.compute_evacuation_route("Room101")
print(f"Route after clearing: {route2}", flush=True)
