import random
from collections import deque
from typing import Dict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import generate_full_mesh
import paralelize
import copy
import math

# TODO allow encrypted optimization
# TODO allow path recalculation when we run out of capacity


# ==========================================================
# 1. Basic Network Structure
# ==========================================================
class Network:
    def __init__(self, edges, base_capacity=90, encrypted_cap=30):
        self.base_capacity = base_capacity
        self.encrypted_cap = encrypted_cap
        self.mst = []
        self.edges = {}
        self.nodes = set()
        for u, v in edges:
            self.nodes.update([u, v])
            self.edges[(u, v)] = base_capacity
            self.edges[(v, u)] = base_capacity
        self.shortest_paths = {}

    def neighbors(self, node):
        return [v for (u, v) in self.edges if u == node]

    def reset_capacities(self):
        for e in self.edges:
            self.edges[e] = self.base_capacity
        self.encrypt_edges()

    def encrypt_edges(self):
        for u, v in self.mst:
            self.nodes.update([u, v])
            self.edges[(u, v)] = self.encrypted_cap
            self.edges[(v, u)] = self.encrypted_cap


class Transciever:
    def __init__(self, a_throughput, a_cost) -> None:
        self.throughput = a_throughput
        self.cost = a_cost

    def representation(self) -> str:
        return f"T_{self.throughput}"


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


# ==========================================================
# 4. Chromosome Helpers
# ==========================================================
def all_pairs(nodes):
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]


def chromosome_to_deployments(
    chromosome, nodes, transceivers: Dict[str, Transciever]
):
    pairs = all_pairs(nodes)
    t_types = [t for t in transceivers.keys()]
    deployments = []
    idx = 0
    for src, dst in pairs:
        for t_type in t_types:
            count = chromosome[idx]
            idx += 1
            if count > 0:
                deployments.append((src, dst, t_type, count))
    return deployments


def random_chromosome(nodes, transceivers, max_count=10):
    num_pairs = len(all_pairs(nodes))
    num_genes = num_pairs * len(transceivers)
    return [random.randint(0, max_count) for _ in range(num_genes)]


# ==========================================================
# 7. Evolutionary Algorithm (Mutation-only)
# ==========================================================
def mutate(
    chromosome,
    overflow_flag,
    mutation_rate=0.2,
    max_step=2,
):
    new_chrom = chromosome.copy()
    for i in range(len(new_chrom)):
        if random.random() < mutation_rate:
            step = random.randint(-max_step, max_step)
            new_chrom[i] = max(0, new_chrom[i] + step - overflow_flag)
    return new_chrom


overflowing_indices = set()


def mutate_guided(
    chromosome,
    guidance,
    trans_len,
    possible,
    overlfowing_edge,
    mutation_rate=0.2,
    max_step=2,
):
    # mesh_count = 50
    # print(guidance)
    # if overlfowing_edge is not None:
    #     src, dst = overlfowing_edge
    #     idx = (src * (mesh_count - src) + dst - 1) * trans_len
    #     tmp_idx = 0
    #     overflowing_indices.add(idx)
    #     for k in range(2):
    #         for i in range(trans_len - 1):
    #             if chromosome[idx + i] > 0:
    #                 chromosome[idx + i] = chromosome[idx + i] - 1
    #                 tmp_idx = i
    #                 break
    #     print(
    #         f"Fixed overflowing edge by removing two on {tmp_idx} and adding on {tmp_idx+1}"
    #     )
    #     print(overflowing_indices)
    #     if (tmp_idx + 1) < trans_len:
    #         chromosome[idx + tmp_idx + 1] += 1

    if possible:
        for idx, value in guidance:

            # c_index = random.choices(
            #     range(0, trans_len),
            #     weights=[math.sqrt(1 / (i + 1)) for i in range(0, trans_len)],
            # )
            # [0]
            # TODO fix weights
            c_index = random.randint(0, trans_len - 1)
            # if idx in overflowing_indices:
            #     print(f"{chromosome[idx:idx+trans_len]}")
            # c_index = random.choices(
            #     range(trans_len), cum_weights=(1, 0.9, 0.8, 0.3), k=1
            # )[0]
            index = idx * trans_len + c_index
            if value == -1:
                while (
                    chromosome[index] == 0
                    and index < (idx + 1) * trans_len - 1
                ):
                    index += 1
            chromosome[index] = max(chromosome[index] + value, 0)
        # Cost saving
        for idx in range(0, len(chromosome), trans_len):
            if (
                idx not in overflowing_indices
                and random.random() < mutation_rate
            ):
                for i in range(trans_len):
                    if chromosome[idx + trans_len - i - 1] > 0:
                        chromosome[idx + trans_len - i - 1] -= 1
                        chromosome[idx + trans_len - i - 2] += 2
                        break
    else:
        # print(chromosome)
        for idx in range(0, len(chromosome), trans_len):
            tmp_idx = 0
            if random.random() < 0.3:
                for k in range(2):
                    # c_index = random.randint(0, trans_len - 1)
                    # index = idx + c_index
                    # chromosome[index] = max(chromosome[index] - 1, 0)
                    for i in range(trans_len):
                        if chromosome[idx + i] > 0:
                            chromosome[idx + i] = chromosome[idx + i] - 1
                            tmp_idx = i
                            break
                if (tmp_idx + 1) < trans_len:
                    chromosome[min(idx + tmp_idx + 1, idx + trans_len - 1)] = (
                        chromosome[min(idx + tmp_idx + 1, idx + trans_len - 1)]
                        + 1
                    )
            # if i == 3:
            #     chromosome[(idx + 1) * trans_len - 1] = 2

    return chromosome


def tournament_selection(
    pop,
    fitnesses,
    guidance,
    possible,
    k=10,
):
    selected = random.sample(list(zip(pop, fitnesses, guidance, possible)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return (selected[0][0], selected[0][2], selected[0][3])


def single_point_crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be of the same length")

    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def evolutionary_algorithm(
    net,
    nodes,
    transceivers,
    per_node_demand,
    population_size=10,
    generations=50,
):
    try:
        pop = [
            random_chromosome(nodes, transceivers, max_count=7)
            for _ in range(population_size)
        ]
        best_overall, best_fitness = None, float("-inf")
        for gen in range(generations):

            fitnesses = paralelize.appraise_pop(
                net,
                pop,
                per_node_demand,
                [(40, 1), (100, 3), (500, 20), (1000, 50)],
                -500000000,
                -1,
                -50000,
                -1000000,
                -5,
                100,
            )

            gen_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i][0])
            if fitnesses[gen_best_idx][0] > best_fitness:
                best_fitness = fitnesses[gen_best_idx][0]
                best_overall = pop[gen_best_idx]

            print(
                f"Gen {gen+1:03}: "
                f"Fitness={fitnesses[gen_best_idx][0]:.2f}, "
                f"Unmet={fitnesses[gen_best_idx][2]}, "
                f"Unconnected_nodes={fitnesses[gen_best_idx][1]}, ",
                f"possible={fitnesses[gen_best_idx][3]}, ",
                f"overflow={fitnesses[gen_best_idx][4]}",
                f"overflowing_edge={fitnesses[gen_best_idx][6]}",
            )
            guidance = [x[5] for x in fitnesses]
            possible = [x[3] for x in fitnesses]
            # mut_rate = 0.05 if fitnesses[gen_best_idx][2] > 30000 else 0.5
            new_pop = []
            for _ in range(population_size):
                parent = tournament_selection(
                    pop, fitnesses, guidance, possible, k=1
                )
                child = mutate_guided(
                    parent[0],
                    parent[1],
                    len(transceivers),
                    parent[2],
                    fitnesses[gen_best_idx][6],
                    # mutation_rate=mut_rate,
                )
                new_pop.append(child)
            pop = new_pop
    except KeyboardInterrupt:
        print(best_overall)
    return best_overall, best_fitness


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_network(
    edges,
    edge_usage,
    base_capacity,
    encrypted_cap,
    mst_edges,
    title="Network Usage (used/capacity)",
    filename="network.png",
):
    """
    Visualize the network from the Rust `Network` struct exposed via PyO3.
    Works with:
        - net.base_capacity (u32)
        - net.encrypted_cap (u32)
        - net.mst (list of (u32, u32))
        - net.edges (dict[(u32, u32)] = int)
    """
    G = nx.Graph()

    # Convert MST to a Python set for faster lookup
    print(mst_edges)
    mst_edges = set(map(tuple, mst_edges))

    for u, v in edges:
        # Determine capacity: encrypted if edge in MST
        if (u, v) in mst_edges or (v, u) in mst_edges:
            cap = encrypted_cap
        else:
            cap = base_capacity

        # Get the current edge usage (stored value)
        edge_value = edge_usage.get((u, v)) or edge_usage.get((v, u))
        if edge_value is None:
            continue  # Skip if missing

        # Usage = how much of capacity is consumed (capacity - remaining)
        usage = cap - edge_value
        G.add_edge(u, v, capacity=cap, usage=usage)

    # Layout and figure
    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Compute usage ratio
    usages = [G[u][v]["usage"] / G[u][v]["capacity"] for u, v in G.edges()]
    colors = cm.viridis(usages)

    # Draw nodes and edges
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


def run_ev():
    # edges = [
    #     ("A", "B"),
    #     ("B", "C"),
    #     ("C", "D"),
    #     ("A", "D"),
    #     ("B", "D"),
    #     ("E", "D"),
    #     ("E", "F"),
    #     ("A", "F"),
    # ]
    # nodes = ["A", "B", "C", "D", "E", "F"]
    edges, nodes = generate_full_mesh(50)
    net = Network(edges)
    shortest_paths = {}
    transceivers = [
        Transciever(40, 1),
        Transciever(100, 5),
        Transciever(500, 20),
        Transciever(1000, 50),
    ]
    transceivers = {t.representation(): t for t in transceivers}
    per_node_demand = 80000
    net.mst = fast_kruskal_mst(net)
    net.encrypt_edges()
    print(net.mst)
    chrom = random_chromosome(nodes, transceivers)
    print(len(chrom))
    new_net = paralelize.Network(90, 30, edges, nodes)
    new_net.set_shortest_paths()
    new_net.set_mst(net.mst)
    new_net.reset_capacities()
    best_chrom, best_fit = evolutionary_algorithm(
        new_net,
        nodes,
        transceivers,
        per_node_demand,
        population_size=1,
        generations=5000,
    )

    print("\nBest chromosome found:", best_chrom)
    best_deployments = chromosome_to_deployments(
        best_chrom, nodes, transceivers
    )
    # for d in best_deployments:
    #     print(" ", d)

    # result = fitness_for_deployments(
    #     new_net, transceivers, best_chrom, per_node_demand, nodes
    # )
    # print("\nFitness result:", result)
    transceivers = [(40, 1), (100, 3), (500, 20), (1000, 50)]
    new_edges = new_net.get_edge_use(best_chrom, transceivers)
    visualize_network(edges, new_edges, 90, 30, new_net.mst)
    # net.reset_capacities()
    # print(net.edges)
    print("Network visualization saved as 'network.png'")


def climb():
    edges, nodes = generate_full_mesh(5)
    transceivers = [(40, 1), (100, 3), (500, 20), (1000, 50)]
    chrom = random_chromosome(nodes, transceivers)
    net = paralelize.Network(90, 30, edges, nodes)
    net.set_shortest_paths()
    iters = 1000
    per_node_demand = 1500
    for iter in range(iters):
        pop = get_neighbours(
            chrom, iter % (len(chrom) // len(transceivers)), transceivers
        )
        fitnesses = paralelize.appraise_pop(
            net,
            pop,
            per_node_demand,
            [(40, 1), (100, 3), (500, 20), (1000, 50)],
            -500000,
            -1000000,
            -50000,
            -0.1,
        )
        gen_best_idx = max(range(len(pop)), key=lambda i: fitnesses[i][0])
        chrom = pop[gen_best_idx]
        print(
            f"Gen {iter+1:03}: "
            f"Fitness={fitnesses[gen_best_idx][0]:.2f}, "
            f"Unmet={fitnesses[gen_best_idx][2]}, "
            f"Unconnected_nodes={fitnesses[gen_best_idx][1]}, ",
            f"possible={fitnesses[gen_best_idx][3]}, ",
            f"overflow={fitnesses[gen_best_idx][4]}",
        )
    print(chrom)
    new_edges = net.get_edge_use(chrom, transceivers)
    visualize_network(edges, new_edges, 90)


def get_neighbours(chrom, idx, transceivers):
    neighbours = []
    for i in range(len(transceivers)):
        for j in range(10):
            new_chrom1 = copy.deepcopy(chrom)
            new_chrom2 = copy.deepcopy(chrom)
            new_chrom1[idx + i] += j
            if new_chrom2[idx + i] > j:
                new_chrom2[idx + i] -= j

            neighbours.append(new_chrom1)
            neighbours.append(new_chrom2)
    return neighbours


# ==========================================================
# 9. Run Example
# ==========================================================
if __name__ == "__main__":
    run_ev()
    # climb()
