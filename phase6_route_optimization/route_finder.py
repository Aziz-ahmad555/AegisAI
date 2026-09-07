import networkx as nx
from building_graph import build_building_graph, visualize_graph

def find_evacuation_route(G, start, exit_options, blocked_edges=None):
    G_working = G.copy()

    if blocked_edges:
        for u, v in blocked_edges:
            if G_working.has_edge(u, v):
                G_working.remove_edge(u, v)

    best_path = None
    best_length = float("inf")
    best_exit = None

    for exit_node in exit_options:
        try:
            length = nx.dijkstra_path_length(G_working, start, exit_node, weight="weight")
            if length < best_length:
                best_length = length
                best_path = nx.dijkstra_path(G_working, start, exit_node, weight="weight")
                best_exit = exit_node
        except nx.NetworkXNoPath:
            continue

    return best_path, best_length, best_exit


if __name__ == "__main__":
    G = build_building_graph()
    exits = ["ExitMain", "ExitEmergency"]
    start_room = "Room101"

    print(f"=== Scenario 1: Normal conditions, evacuating from {start_room} ===")
    path, length, exit_used = find_evacuation_route(G, start_room, exits)
    print(f"Best route: {' -> '.join(path)}")
    print(f"Total distance: {length}")
    print(f"Exit used: {exit_used}\n")
    visualize_graph(G, path=path, save_path="route_normal.png")

    print(f"=== Scenario 2: Fire blocks the route actually being used (CorridorA <-> ExitEmergency) ===")
    blocked = [("CorridorA", "ExitEmergency")]
    path2, length2, exit_used2 = find_evacuation_route(G, start_room, exits, blocked_edges=blocked)
    print(f"Rerouted path: {' -> '.join(path2)}")
    print(f"Total distance: {length2}")
    print(f"Exit used: {exit_used2}\n")
    visualize_graph(G, path=path2, blocked_edges=blocked, save_path="route_blocked.png")

    print(f"=== Scenario 3: Even more severe - CorridorA is fully cut off from CorridorB and ExitEmergency ===")
    blocked_severe = [("CorridorA", "ExitEmergency"), ("CorridorA", "CorridorB")]
    path3, length3, exit_used3 = find_evacuation_route(G, start_room, exits, blocked_edges=blocked_severe)
    if path3:
        print(f"Rerouted path: {' -> '.join(path3)}")
        print(f"Total distance: {length3}")
        print(f"Exit used: {exit_used3}\n")
        visualize_graph(G, path=path3, blocked_edges=blocked_severe, save_path="route_severe.png")
    else:
        print("No path found - Room101 is completely cut off from all exits!\n")
