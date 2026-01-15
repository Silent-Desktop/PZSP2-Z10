from jaxtyping import jaxtyped
from beartype import beartype
import torch
from jaxtyping import Float
from torch import Tensor

from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint


class PathEdgeBandwidthKirchhoffMatch(Constraint):
    readable_name: str = "Node In == Out"

    @jaxtyped(typechecker=beartype)
    def check(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ) -> Float[Tensor, "P"]:
        return self._mean_similarity_scores(
            self._check_all(population, transponder_capacities, demand)
        )

    @jaxtyped(typechecker=beartype)
    def _check_all(
        self,
        population: Population,
        transponder_capacities: Float[Tensor, "T"],
        demand: Float[Tensor, "N N"],
    ) -> Float[Tensor, "P N N N"]:
        p = population.path_edge_bandwidth_usage
        N = population.encrypted_neigh_matrix.size(1)
        node_input_sum = torch.einsum(
            "pixkl -> pxkl", p
        )  # (P, N, N, N, N) -> (P, N, N, N) for each node on each path
        node_output_sum = torch.einsum(
            "pxjkl -> pxkl", p
        )  # (P, N, N, N, N) -> (P, N, N, N) for each node on each path
        # check that for each node on each path input and output matches
        # path root and dest (k,l) should be excluded
        identity_mask = torch.eye(N, dtype=torch.bool)
        mask_x_eq_k = identity_mask.view(1, N, N, 1)
        mask_x_eq_l = identity_mask.view(1, N, 1, N)
        final_mask = mask_x_eq_k | mask_x_eq_l
        node_input_sum[final_mask.expand_as(node_input_sum)] = 0.0
        node_output_sum[final_mask.expand_as(node_output_sum)] = 0.0
        return self._calculate_similarity_scores_all(
            node_input_sum, node_output_sum
        )
