from pydantic import BaseModel
from pydantic.config import ConfigDict
import torch
from torch.distributions.bernoulli import Bernoulli

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

    def mutate(self, population : Population, elite_size: int) -> Population:
        elite_paths = population.path_edge_bandwidth_usage[:elite_size]
        elite_transponders = population.path_transponder_assignment[:elite_size]

        non_elite_paths = population.path_edge_bandwidth_usage[elite_size:]
        non_elite_transponders = population.path_transponder_assignment[elite_size:]

        paths_mut_idx = Bernoulli(self.path_edge_bandwidth_usage_mut_proba)\
            .sample(non_elite_paths.size())
        transponders_mut_idx = Bernoulli(self.path_transponder_assignment_mut_proba)\
            .sample(non_elite_transponders.size())
        
        #TODO?: Masking with idx to not calculate random for 0 indices
        random_paths = torch.poisson(non_elite_paths).clamp(max=population.edge_size_limits)
        random_transponders = torch.poisson(non_elite_transponders)
   
        mutated_paths = (1 -  paths_mut_idx) * non_elite_paths +\
                                        paths_mut_idx * random_paths
        mutated_transponders = (1 -  transponders_mut_idx) * non_elite_transponders +\
                                        transponders_mut_idx * random_transponders
        
        new_paths = torch.cat((elite_paths, mutated_paths), dim=0)
        new_transponders = torch.cat((elite_transponders, mutated_transponders), dim=0)
        
        
        return Population.masked(
            encrypted_neigh_matrix=population.encrypted_neigh_matrix,
            path_edge_bandwidth_usage=new_paths,
            path_transponder_assignment=new_transponders
        )