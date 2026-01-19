from jaxtyping import jaxtyped
from beartype import beartype
import torch
from jaxtyping import Float
from torch import Tensor

from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint


class BandwidthLimit(Constraint):
    readable_name: str = "Bandwidth Limit"

    @jaxtyped(typechecker=beartype)
    def check(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ) -> Float[Tensor, "P"]:
        edge_size_useage, pop_edge_size_limits = self._calc(
            population, transponder_capacities, demand
        )
        # relu to cut negative scores. If the limit >= useage -> score = 0
        return self._calculate_similarity_scores_from_sums(
            edge_size_useage, pop_edge_size_limits, diff_transform=torch.relu
        )

    @jaxtyped(typechecker=beartype)
    def _check_all(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ) -> Float[Tensor, "P ..."]:
        edge_size_useage, pop_edge_size_limits = self._calc(
            population, transponder_capacities, demand
        )
        # relu to cut negative scores. If the limit >= useage -> score = 0
        return self._calculate_similarity_scores_all(
            edge_size_useage, pop_edge_size_limits, diff_transform=torch.relu
        )

    def _calc(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ):
        e = population.edge_size_limits  # (N, N)
        p = population.path_edge_bandwidth_usage  # (P, N, N, N, N)
        pop_edge_size_limits = e.unsqueeze(0).expand(
            p.size(0), -1, -1
        )  # (P, N, N)
        edge_size_useage = torch.einsum("pijkl -> pij", p)  # (P, N, N)
        edge_size_useage += edge_size_useage.tril(diagonal=-1).mT
        return edge_size_useage.triu_(diagonal=1), pop_edge_size_limits
