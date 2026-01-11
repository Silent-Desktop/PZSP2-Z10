from jaxtyping import jaxtyped
from beartype import beartype
import torch
from jaxtyping import Float
from torch import Tensor

from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint

class PathTransponderBandwidthMatch(Constraint):
    readable_name: str = "Path Transponder Bandwidth Match"
    @jaxtyped(typechecker=beartype)
    def check(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        return self._mean_similarity_scores(self._check_all(population, transponder_capacities, demand))

    @jaxtyped(typechecker=beartype)
    def _check_all(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P N N"]:
        p = population.path_edge_bandwidth_usage
        t = population.path_transponder_assignment
        # just check for source because PathBandwidthIOMatch ensures IO equality
        source_output_sum = torch.einsum('pijil -> pij', p) # (P, N, N, N, N) -> (P, N, N)
        transponder_on_path_count = t.sum(dim=3) # (P, N, N, T) -> (P, N, N)
        return self._calculate_similarity_scores_all(source_output_sum, transponder_on_path_count)