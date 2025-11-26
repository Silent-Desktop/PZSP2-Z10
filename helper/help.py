import random
from collections import deque
from typing import Dict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Network:
    def __init__(self, edges, base_capacity=30):
        self.base_capacity = base_capacity
        self.edges = {}
        self.nodes = set()
        for u, v in edges:
            self.nodes.update([u, v])
            self.edges[(u, v)] = base_capacity
            self.edges[(v, u)] = base_capacity

    def neighbors(self, node):
        return [v for (u, v) in self.edges if u == node]

    def reset_capacities(self):
        for e in self.edges:
            self.edges[e] = self.base_capacity


def shortest_path(network, start, goal):
    """
    BFS shortest path that only uses edges with capacity > 0.
    Returns list of nodes or None if no path exists.
    """
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in network.neighbors(node):
            # only consider edges with remaining capacity > 0
            if neighbor not in visited and network.edges[(node, neighbor)] > 0:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def all_pairs(nodes):
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]


def chromosome_to_deployments(chromosome, nodes, transceivers):
    pairs = all_pairs(nodes)
    t_types = list(transceivers.keys())
    deployments = []
    idx = 0
    for src, dst in pairs:
        for t_type in t_types:
            count = chromosome[idx]
            idx += 1
            if count > 0:
                deployments.append((src, dst, t_type, count))
    return deployments


def apply_transcievers(net, nodes, deployment):
    for src, dst, t_type, count in deployment:
        path = shortest_path(net, src, dst)
        if not path:
            return False
        for _ in range(count):
            for u, v in zip(path[:-1], path[1:]):
                net.edges[(u, v)] -= 1
                net.edges[(v, u)] -= 1
                # Recalculate path when run out of capacity
                if net.edges[(u, v)] == 0:
                    path = shortest_path(net, src, dst)
                    if not path:
                        return False
    return True


def kruskal_mst(network):
    """
    Compute the Minimum Spanning Tree (MST) of the network using Kruskal's algorithm.
    Returns a list of edges (u, v) that form the MST and the total cost.
    """
    # Sort edges by weight (lower cost = better)
    # We treat cost as inverse of capacity, so higher capacity = lower cost
    unique_edges = {}
    for (u, v), cap in network.edges.items():
        key = tuple(sorted((u, v)))
        if key not in unique_edges:
            unique_edges[key] = cap

    sorted_edges = sorted(
        unique_edges.items(), key=lambda e: 1 / e[1]
    )  # sort by inverse capacity

    # Union-Find (Disjoint Set Union)
    parent = {node: node for node in network.nodes}
    rank = {node: 0 for node in network.nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return False
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True

    mst_edges = []
    total_cost = 0.0

    for (u, v), cap in sorted_edges:
        if union(u, v):
            mst_edges.append((u, v))
            total_cost += 1 / cap  # inverse capacity as cost
        # Stop when MST is complete
        if len(mst_edges) == len(network.nodes) - 1:
            break

    return mst_edges, total_cost


def visualize_network(
    edges, net, title="Network Usage (used/capacity)", filename="network.png"
):
    G = nx.Graph()
    for u, v in edges:
        cap = net.base_capacity
        usage = cap - net.edges[(u, v)]
        G.add_edge(u, v, capacity=cap, usage=usage)

    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Compute usage ratio for coloring
    usages = [G[u][v]["usage"] / G[u][v]["capacity"] for u, v in G.edges()]
    colors = cm.viridis(usages)

    # Draw network
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color="skyblue",
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    edge_widths = [2 + 5 * u for u in usages]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=colors, ax=ax)

    # Edge labels (usage / capacity)
    edge_labels = {
        (u, v): f"{G[u][v]['usage']}/{G[u][v]['capacity']}"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.6, ax=ax
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Usage ratio", rotation=270, labelpad=15)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def fast_kruskal_mst(network):
    """
    Extremely fast Kruskal-style MST for unweighted networks.
    Uses no sorting, minimal tuple creation, and tight union–find loops.
    """
    nodes = list(network.nodes)
    n = len(nodes)
    node_index = {
        node: i for i, node in enumerate(nodes)
    }  # avoid hash lookups on strings

    parent = list(range(n))
    rank = [0] * n

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1
        return True

    # Build adjacency once to skip duplicate edges
    seen = set()
    mst_edges = []
    need = n - 1

    for u, v in network.edges.keys():
        if (v, u) in seen:
            continue
        seen.add((u, v))
        ui = node_index[u]
        vi = node_index[v]
        if union(ui, vi):
            mst_edges.append((u, v))
            if len(mst_edges) == need:
                break

    return mst_edges


if __name__ == "__main__":
    edges_start = [
        ("A", "B"),
        ("A", "D"),
        ("C", "D"),
        ("D", "E"),
        ("C", "A"),
        ("C", "E"),
        ("B", "E"),
    ]
    net = Network(edges_start, base_capacity=30)
    nodes = ["A", "B", "C", "D"]
    path = [
        ("A", "C"),
        ("C", "D"),
        ("A", "D"),
        ("D", "E"),
    ]
    edges = fast_kruskal_mst(net)
    print(edges)
    visualize_network(edges, net, filename="mst.png")
    visualize_network(edges_start, net)
