from pydantic import BaseModel
from pydantic.config import ConfigDict
import torch
from torch.distributions.bernoulli import Bernoulli
from torch import Tensor
from jaxtyping import Bool

from net_opt.core.population import Population
from net_opt.core.mutations.base_mutation import Mutation


class GeneMutation(Mutation, BaseModel):
    """
    Applies element-wise "gene" mutation to non-elite individuals.

    For each element (gene) in each tensor, a Bernoulli trial determines
    whether it should be mutated, based on component-specific probabilities.

    The mutation logic is as follows:
    - Paths (float):   New = (1 - X_P) * Old + X_P * Random_P
    - Transp. (float): New = (1 - X_T) * Old + X_T * Random_T

    Where:
    - X is a mutation mask (0 or 1) from Bernoulli(p_mut).
    - Random is a new value sampled from a given distribution.
    - neigh_matrix is a global mask to ensure graph validity.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    path_edge_bandwidth_usage_mut_proba: float
    path_transponder_assignment_mut_proba: float

    def mutate(
        self,
        population: Population,
        elite_size: int,
        neigh_matrix: Bool[Tensor, "N N"],
    ) -> Population:
        elite_paths = population.path_edge_bandwidth_usage[:elite_size]
        elite_transponders = population.path_transponder_assignment[
            :elite_size
        ]

        non_elite_paths = population.path_edge_bandwidth_usage[elite_size:]
        non_elite_transponders = population.path_transponder_assignment[
            elite_size:
        ]

        paths_mut_idx = (
            Bernoulli(self.path_edge_bandwidth_usage_mut_proba)
            .sample(non_elite_paths.size())
            .triu_(diagonal=1)
            .bool()
            & neigh_matrix[None, :, :, None, None]
        )
        transponders_mut_idx = (
            Bernoulli(self.path_transponder_assignment_mut_proba)
            .sample(non_elite_transponders.size())
            .triu_(diagonal=1)
            .bool()
        )

        non_elite_paths[paths_mut_idx] = torch.poisson(
            non_elite_paths[paths_mut_idx]
        ).clamp_(
            max=population.edge_size_limits[None, :, :, None, None].expand_as(
                non_elite_paths
            )[paths_mut_idx]
        )
        non_elite_transponders[transponders_mut_idx] = torch.poisson(
            non_elite_transponders[transponders_mut_idx]
        )

        new_paths = torch.cat((elite_paths, non_elite_paths), dim=0)
        new_transponders = torch.cat(
            (elite_transponders, non_elite_transponders), dim=0
        )

        return Population.masked(
            encrypted_neigh_matrix=population.encrypted_neigh_matrix,
            path_edge_bandwidth_usage=new_paths,
            path_transponder_assignment=new_transponders,
            neigh_matrix=neigh_matrix,
            encrypted_bandwidth=population.encrypted_bandwidth,
            regular_bandwidth=population.regular_bandwidth,
        )
