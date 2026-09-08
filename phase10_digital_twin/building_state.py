import networkx as nx
import time
import threading


class BuildingDigitalTwin:
    def __init__(self):
        self.graph = self._build_graph()
        self.zone_status = {node: "SAFE" for node in self.graph.nodes()}
        self.zone_risk = {node: 0.0 for node in self.graph.nodes()}
        self.exits = ["ExitMain", "ExitEmergency"]
        self.lock = threading.RLock()
        self.event_log = []

    def _build_graph(self):
        G = nx.Graph()
        nodes = {
            "Room101": (0, 0), "Room102": (0, 2), "Room103": (0, 4),
            "CorridorA": (2, 2), "CorridorB": (4, 2),
            "Room201": (6, 0), "Room202": (6, 4),
            "Stairwell": (4, 4),
            "ExitMain": (6, 2), "ExitEmergency": (2, 4),
        }
        for name, pos in nodes.items():
            G.add_node(name, pos=pos)

        edges = [
            ("Room101", "CorridorA", 2), ("Room102", "CorridorA", 1),
            ("Room103", "ExitEmergency", 1.5), ("CorridorA", "CorridorB", 2),
            ("CorridorA", "ExitEmergency", 2), ("CorridorB", "Room201", 2),
            ("CorridorB", "ExitMain", 1), ("CorridorB", "Stairwell", 1.5),
            ("Stairwell", "Room202", 1), ("Stairwell", "ExitEmergency", 2),
            ("Room202", "ExitMain", 2),
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    def log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.event_log.append(entry)
        if len(self.event_log) > 20:
            self.event_log.pop(0)
        print(entry)

    def start_fire(self, zone):
        with self.lock:
            if zone not in self.graph.nodes():
                return False
            self.zone_status[zone] = "FIRE"
            self.zone_risk[zone] = 85.0
            self.log_event(f"FIRE detected in {zone}")
            return True

    def clear_zone(self, zone):
        with self.lock:
            if zone not in self.graph.nodes():
                return False
            self.zone_status[zone] = "SAFE"
            self.zone_risk[zone] = 0.0
            self.log_event(f"{zone} confirmed clear")
            return True

    def get_blocked_edges(self):
        with self.lock:
            blocked = []
            for u, v in self.graph.edges():
                if self.zone_status.get(u) == "FIRE" or self.zone_status.get(v) == "FIRE":
                    blocked.append((u, v))
            return blocked

    def compute_evacuation_route(self, start):
        with self.lock:
            G_working = self.graph.copy()
            for u, v in self.get_blocked_edges():
                if G_working.has_edge(u, v):
                    G_working.remove_edge(u, v)

            best_path, best_len, best_exit = None, float("inf"), None
            for exit_node in self.exits:
                try:
                    length = nx.dijkstra_path_length(G_working, start, exit_node, weight="weight")
                    if length < best_len:
                        best_len = length
                        best_path = nx.dijkstra_path(G_working, start, exit_node, weight="weight")
                        best_exit = exit_node
                except nx.NetworkXNoPath:
                    continue

            return {"path": best_path, "length": best_len if best_path else None, "exit": best_exit}

    def get_overall_risk(self):
        with self.lock:
            if not self.zone_risk:
                return 0.0
            return round(sum(self.zone_risk.values()) / len(self.zone_risk), 1)

    def get_state_snapshot(self):
        with self.lock:
            pos = nx.get_node_attributes(self.graph, "pos")
            return {
                "nodes": [
                    {"id": n, "x": pos[n][0], "y": pos[n][1], "status": self.zone_status[n], "risk": self.zone_risk[n]}
                    for n in self.graph.nodes()
                ],
                "edges": [
                    {"from": u, "to": v, "weight": self.graph[u][v]["weight"]}
                    for u, v in self.graph.edges()
                ],
                "blocked_edges": self.get_blocked_edges(),
                "overall_risk": self.get_overall_risk(),
                "event_log": self.event_log[-10:],
            }


if __name__ == "__main__":
    twin = BuildingDigitalTwin()
    print("Initial state - overall risk:", twin.get_overall_risk())

    print("\n--- Scenario 1: Fire in CorridorB (Room101 has an alternate path) ---")
    twin.start_fire("CorridorB")
    route = twin.compute_evacuation_route("Room101")
    print(f"Route from Room101: {route}")
    print(f"Overall risk: {twin.get_overall_risk()}")
    twin.clear_zone("CorridorB")

    print("\n--- Scenario 2: Fire in CorridorA (Room101's only connection - no alternate exists) ---")
    twin.start_fire("CorridorA")
    route2 = twin.compute_evacuation_route("Room101")
    print(f"Route from Room101: {route2}")
    if route2["path"] is None:
        print("  -> Correctly detected: Room101 has no alternate route in this layout")
    print(f"Overall risk: {twin.get_overall_risk()}")
    twin.clear_zone("CorridorA")

    print("\n--- Final check: route restored after clearing ---")
    route3 = twin.compute_evacuation_route("Room101")
    print(f"Route from Room101: {route3}")
