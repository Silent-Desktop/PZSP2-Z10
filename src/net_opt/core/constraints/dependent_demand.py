from jaxtyping import jaxtyped
from beartype import beartype
import torch
from jaxtyping import Float
from torch import Tensor

from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint
from net_opt.core.constraints.path_bandwidth_io_match import PathBandwidthIOMatch
from net_opt.core.constraints.path_transponder_bandwidth_at_least import PathTransponderBandwidthAtLeast
from net_opt.core.constraints.path_edge_bandwidth_kirchhoff import PathEdgeBandwidthKirchhoff

class DependentDemand(Constraint):
    readable_name: str = "Dependent Demand"
    @jaxtyped(typechecker=beartype)
    def check(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        demand_per_individual, path_coverage = self._calc(population, transponder_capacities, demand)
        #relu to cut negative scores. If the coverage >= demand -> score = 0
        return self._calculate_similarity_scores_from_sums(demand_per_individual, path_coverage, diff_transform=torch.relu)
    
    @jaxtyped(typechecker=beartype)
    def _check_all(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P ..."]:
        demand_per_individual, path_coverage = self._calc(population, transponder_capacities, demand)
        #relu to cut negative scores. If the coverage >= demand -> score = 0
        return self._calculate_similarity_scores_all(demand_per_individual, path_coverage, diff_transform=torch.relu)
    
    def _calc(self, population, transponder_capacities, demand):
        t = population.path_transponder_assignment #(P, N, N, T)
        P = t.size(0)
        path_coverage = t @ transponder_capacities # (P, N, N, T) @ (T,) -> (P, N, N)
        # actual demand coverage is dependent on the amount of complete paths:
        real_paths = PathBandwidthIOMatch()._check_all(population, transponder_capacities, demand)\
					* PathTransponderBandwidthAtLeast()._check_all(population, transponder_capacities, demand)\
					* PathEdgeBandwidthKirchhoff()._check_all(population, transponder_capacities, demand).flatten(start_dim=3).mean(dim=3)
        path_coverage *= real_paths
        demand_per_individual = demand.unsqueeze(0).expand(P, -1, -1)
        return demand_per_individual, path_coverage