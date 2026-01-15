from jaxtyping import Float
from torch import Tensor
from net_opt.core.population import Population

from pydantic import BaseModel
import torch

from net_opt.core.selections.base_selection import Selection


class TournamentSelection(Selection, BaseModel):
    k: int

    def get_next_generation(
        self,
        population: Population,
        penalties: Float[Tensor, "P"],
        elite_size: int,
    ) -> Population:
        elite_paths = population.path_edge_bandwidth_usage[:elite_size]
        elite_transponders = population.path_transponder_assignment[
            :elite_size
        ]

        population_size = penalties.size(0)
        tournaments_idx = torch.randint(
            low=0,
            high=population_size,
            size=(population_size - elite_size, self.k),
        )  # (P-elite_size, k)

        tournament_penalties = penalties[tournaments_idx]  # (P-elite_size, k)
        winner_local_indices = torch.min(
            tournament_penalties, dim=1
        ).indices  # (P-elite_size)
        row_indices = torch.arange(population_size - elite_size)
        final_winner_indices = tournaments_idx[
            row_indices, winner_local_indices
        ]

        selected_paths = population.path_edge_bandwidth_usage[
            final_winner_indices
        ]
        selected_transponders = population.path_transponder_assignment[
            final_winner_indices
        ]

        new_paths = torch.cat((elite_paths, selected_paths), dim=0)
        new_transponders = torch.cat(
            (elite_transponders, selected_transponders), dim=0
        )

        return Population(
            encrypted_neigh_matrix=population.encrypted_neigh_matrix,
            path_edge_bandwidth_usage=new_paths,
            path_transponder_assignment=new_transponders,
            encrypted_bandwidth=population.encrypted_bandwidth,
            regular_bandwidth=population.regular_bandwidth
        )
