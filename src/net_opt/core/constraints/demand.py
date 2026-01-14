from jaxtyping import jaxtyped
from beartype import beartype
import torch
from jaxtyping import Float
from torch import Tensor

from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint


class Demand(Constraint):
    readable_name: str = "Demand"

    @jaxtyped(typechecker=beartype)
    def check(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ) -> Float[Tensor, "P"]:
        demand_per_individual, path_coverage = self._calc(
            population, transponder_capacities, demand
        )
        # relu to cut negative scores. If the coverage >= demand -> score = 0
        return self._calculate_similarity_scores_from_sums(
            demand_per_individual, path_coverage, diff_transform=torch.relu
        )

    @jaxtyped(typechecker=beartype)
    def _check_all(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ) -> Float[Tensor, "P ..."]:
        demand_per_individual, path_coverage = self._calc(
            population, transponder_capacities, demand
        )
        # relu to cut negative scores. If the coverage >= demand -> score = 0
        return self._calculate_similarity_scores_all(
            demand_per_individual, path_coverage, diff_transform=torch.relu
        )

    def _calc(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ):
        t = population.path_transponder_assignment  # (P, T, N, N)
        P = t.size(0)
        path_coverage = (
            t.permute(0, 2, 3, 1) @ transponder_capacities
        )  # (P, N, N, T) @ (T,) -> (P, N, N)
        demand_per_individual = demand.unsqueeze(0).expand(P, -1, -1)
        return demand_per_individual, path_coverage
