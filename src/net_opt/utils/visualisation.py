from typing import no_type_check, Optional
import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Bool, Float
from torch import Tensor
import torch

from net_opt.core.population import Population


@no_type_check
def visualize_population_individual(
    population: Population,
    transponder_capacities: Float[Tensor, "T"],
    individual_index: int = 0,
    iteration: Optional[int] = None,
):
    if individual_index >= len(population.path_edge_bandwidth_usage):
        print(f"Error: individual_index {individual_index} out of range.")
        return

    e_matrix = population.encrypted_neigh_matrix
    p_usage = population.path_edge_bandwidth_usage[individual_index]
    t_assign = population.path_transponder_assignment[individual_index]

    edge_bw_usage = torch.einsum("ijkl -> ij", p_usage)
    edge_bw_usage += edge_bw_usage.tril(diagonal=-1).mT
    edge_bw_usage.triu_(diagonal=1)

    # (T, N, N) -> (N, N, T) @ (T) -> (N, N)
    path_capacity = t_assign.permute(1, 2, 0) @ transponder_capacities

    total_trans_usage = t_assign.sum(dim=(1, 2))

    data_to_plot = [
        e_matrix,
        edge_bw_usage,
        path_capacity,
        total_trans_usage,
        transponder_capacities,
    ]
    e_np, edge_np, path_np, trans_np, caps_np = [
        d.cpu().numpy() for d in data_to_plot
    ]

    N = e_np.shape[0]
    T = caps_np.shape[0]
    node_labels = [f"Node {i}" for i in range(N)]
    trans_labels = [f"Type {i}" for i in range(T)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = f"Visualization for Individual {individual_index}"
    if iteration is not None:
        title += f", iteration {iteration}"
    fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    ax.imshow(e_np, cmap="Greys", interpolation="nearest")
    ax.set_title("Encrypted Adjacency Matrix (Black = Encrypted)")
    ax.set_xticks(np.arange(N), node_labels, rotation="vertical")
    ax.set_yticks(np.arange(N), node_labels)

    for i in range(N):
        for j in range(N):
            text_val = int(e_np[i, j])
            color = "white" if text_val == 1 else "black"
            ax.text(j, i, text_val, ha="center", va="center", color=color)

    ax = axes[0, 1]
    cax2 = ax.imshow(edge_np, cmap="viridis", interpolation="nearest", vmin=0)
    ax.set_title("Total Edge Bandwidth Usage (on physical edges)")
    ax.set_xticks(np.arange(N), node_labels, rotation="vertical")
    ax.set_yticks(np.arange(N), node_labels)
    cbar = fig.colorbar(
        cax2, ax=ax, orientation="vertical", label="Total Bandwidth Used"
    )
    max_val = edge_np.max()
    ticks = np.linspace(0, max_val, 9, dtype=int)
    cbar.set_ticks(ticks)

    ax = axes[1, 0]
    cax3 = ax.imshow(path_np, cmap="plasma", interpolation="nearest", vmin=0)
    ax.set_title("Effective Path Capacity (per path i->j)")
    ax.set_xticks(np.arange(N), node_labels, rotation="vertical")
    ax.set_yticks(np.arange(N), node_labels)
    cbar = fig.colorbar(
        cax3, ax=ax, orientation="vertical", label="Effective Capacity"
    )
    max_val = path_np.max()
    ticks = np.linspace(0, max_val, 9, dtype=int)
    cbar.set_ticks(ticks)

    ax = axes[1, 1]
    bars4 = ax.bar(trans_labels, trans_np, color="coral")
    ax.set_title("Total Transponder Bandwidth Assigned")
    ax.set_ylabel("Total Bandwidth")
    ax.set_xlabel("Transponder Type")
    ax.bar_label(bars4, fmt="%.1f")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


@no_type_check
def visualize_input_data(
    neigh_matrix: Bool[Tensor, "N N"],
    demand: Float[Tensor, "N N"],
    transponder_costs: Float[Tensor, "T"],
    transponder_capacities: Float[Tensor, "T"],
):
    mat_np = neigh_matrix.cpu().numpy()
    dem_np = demand.cpu().numpy()
    costs_np = transponder_costs.cpu().numpy()
    caps_np = transponder_capacities.cpu().numpy()

    N = mat_np.shape[0]
    T = costs_np.shape[0]
    node_labels = [f"Node {i}" for i in range(N)]
    trans_labels = [f"Type {i}" for i in range(T)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("EA Input Data Visualization", fontsize=16)

    ax = axes[0, 0]
    ax.imshow(mat_np, cmap="Greys", interpolation="nearest")
    ax.set_title("Neighborhood Matrix (Black = Edge Exists)")
    ax.set_xticks(np.arange(N), node_labels, rotation="vertical")
    ax.set_yticks(np.arange(N), node_labels)

    ax = axes[0, 1]
    cax2 = ax.imshow(dem_np, cmap="plasma", interpolation="nearest", vmin=0)
    ax.set_title("Demand Matrix (Bandwidth)")
    ax.set_xticks(np.arange(N), node_labels, rotation="vertical")
    ax.set_yticks(np.arange(N), node_labels)

    cbar = fig.colorbar(
        cax2, ax=ax, orientation="vertical", label="Bandwidth Demand"
    )
    max_val = dem_np.max()
    ticks = np.linspace(0, max_val, 9, dtype=int)
    cbar.set_ticks(ticks)

    ax = axes[1, 0]
    bars3 = ax.bar(trans_labels, costs_np, color="skyblue")
    ax.set_title("Transponder Costs")
    ax.set_ylabel("Cost ($)")
    ax.set_xlabel("Transponder Type")
    ax.bar_label(bars3, fmt="$%.0f")

    ax = axes[1, 1]
    bars4 = ax.bar(trans_labels, caps_np, color="lightgreen")
    ax.set_title("Transponder Capacities")
    ax.set_ylabel("Capacity (Gbps)")
    ax.set_xlabel("Transponder Type")
    ax.bar_label(bars4, fmt="%.0f")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
