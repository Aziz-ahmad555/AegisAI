import sys

print("CHECKPOINT 1: script started", flush=True)

import networkx as nx
print("CHECKPOINT 2: networkx imported", flush=True)

import time
import random
import threading
print("CHECKPOINT 3: other imports done", flush=True)


class BuildingDigitalTwin:
    def __init__(self):
        print("CHECKPOINT 4: entering __init__", flush=True)
        self.graph = self._build_graph()
        print("CHECKPOINT 5: graph built", flush=True)
        self.zone_status = {node: "SAFE" for node in self.graph.nodes()}
        self.zone_risk = {node: 0.0 for node in self.graph.nodes()}
        self.exits = ["ExitMain", "ExitEmergency"]
        self.lock = threading.Lock()
        self.event_log = []
        print("CHECKPOINT 6: __init__ complete", flush=True)

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

    def get_state_snapshot(self):
        print("CHECKPOINT 7: entering get_state_snapshot, about to acquire lock", flush=True)
        with self.lock:
            print("CHECKPOINT 8: lock acquired", flush=True)
            pos = nx.get_node_attributes(self.graph, "pos")
            print("CHECKPOINT 9: got positions", flush=True)
            snapshot = {"node_count": len(self.graph.nodes())}
            print("CHECKPOINT 10: snapshot built, returning", flush=True)
            return snapshot


print("CHECKPOINT A: about to create twin", flush=True)
twin = BuildingDigitalTwin()
print("CHECKPOINT B: twin created, about to get snapshot", flush=True)
result = twin.get_state_snapshot()
print("CHECKPOINT C: got snapshot:", result, flush=True)
print("CHECKPOINT D: script finished successfully", flush=True)
