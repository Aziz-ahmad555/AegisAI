import networkx as nx
import matplotlib.pyplot as plt

def build_building_graph():
    G = nx.Graph()

    # Nodes represent rooms/corridors/exits with (x, y) positions for visualization
    nodes = {
        "Room101": (0, 0),
        "Room102": (0, 2),
        "Room103": (0, 4),
        "CorridorA": (2, 2),
        "CorridorB": (4, 2),
        "Room201": (6, 0),
        "Room202": (6, 4),
        "Stairwell": (4, 4),
        "ExitMain": (6, 2),
        "ExitEmergency": (2, 4),
    }

    for name, pos in nodes.items():
        G.add_node(name, pos=pos)

    # Edges represent walkable connections, weight = distance/time cost
    edges = [
        ("Room101", "CorridorA", 2),
        ("Room102", "CorridorA", 1),
        ("Room103", "ExitEmergency", 1.5),
        ("CorridorA", "CorridorB", 2),
        ("CorridorA", "ExitEmergency", 2),
        ("CorridorB", "Room201", 2),
        ("CorridorB", "ExitMain", 1),
        ("CorridorB", "Stairwell", 1.5),
        ("Stairwell", "Room202", 1),
        ("Stairwell", "ExitEmergency", 2),
        ("Room202", "ExitMain", 2),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G


def visualize_graph(G, path=None, blocked_edges=None, save_path="building_graph.png"):
    pos = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(10, 6))

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=8)

    edge_colors = []
    for u, v in G.edges():
        if blocked_edges and ((u, v) in blocked_edges or (v, u) in blocked_edges):
            edge_colors.append("red")
        else:
            edge_colors.append("gray")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="green", width=4)

    plt.title("AegisAI - Building Evacuation Graph")
    plt.axis("off")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    G = build_building_graph()
    print("Building graph created with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges")
    visualize_graph(G)
