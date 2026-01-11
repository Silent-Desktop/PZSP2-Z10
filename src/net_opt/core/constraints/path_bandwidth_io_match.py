from jaxtyping import jaxtyped
from beartype import beartype
import torch
from jaxtyping import Float
from torch import Tensor

from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint


class PathBandwidthIOMatch(Constraint):
    readable_name: str = "Path Bandwidth IO Match"
    @jaxtyped(typechecker=beartype)
    def check(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        return self._mean_similarity_scores(self._check_all(population, transponder_capacities, demand))

    @jaxtyped(typechecker=beartype)
    def _check_all(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P N N"]:
        p = population.path_edge_bandwidth_usage
        source_output_sum = torch.einsum('pijil -> pij', p) # (P, N, N, N, N) -> (P, N, N)
        dest_input_sum = torch.einsum('pijkj -> pij', p) # (P, N, N, N, N) -> (P, N, N)
        return self._calculate_similarity_scores_all(source_output_sum, dest_input_sum)