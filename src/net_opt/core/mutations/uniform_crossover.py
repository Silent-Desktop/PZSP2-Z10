from pydantic import BaseModel
from pydantic.config import ConfigDict
import torch
from torch.distributions.bernoulli import Bernoulli
from jaxtyping import jaxtyped, Bool
from torch import Tensor
from beartype import beartype

from net_opt.core.population import Population
from net_opt.core.mutations.base_mutation import Mutation


class UniformCrossover(Mutation, BaseModel):
    """
    Applies element-wise uniform crossover based on the logic:
    - Child1 = X * Parent1 + (1 - X) * Parent2
    - Child2 = X * Parent2 + (1 - X) * Parent1

    Where X is a tensor of Bernoulli samples (0 or 1).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path_edge_bandwidth_usage_cross_proba: float
    path_transponder_assignment_cross_proba: float

    @jaxtyped(typechecker=beartype)
    def _cross_component(
        self, parent1: Tensor, parent2: Tensor, proba: float
    ) -> tuple[Tensor, Tensor]:
        """
        Performs uniform crossover on a single component (E, P, or T)
        for a batch of parent pairs.
        """
        cross_mask = Bernoulli(proba).sample(parent1.size()).float()
        child1 = cross_mask * parent1 + (1.0 - cross_mask) * parent2
        child2 = cross_mask * parent2 + (1.0 - cross_mask) * parent1
        return child1, child2

    @jaxtyped(typechecker=beartype)
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

        num_to_cross = non_elite_paths.size(0)

        if num_to_cross < 2:
            return population

        indices = torch.randperm(num_to_cross)
        shuffled_paths = non_elite_paths[indices]
        shuffled_transponders = non_elite_transponders[indices]

        num_pairs = num_to_cross // 2

        # first parent
        p1_paths = shuffled_paths[:num_pairs]
        p1_trans = shuffled_transponders[:num_pairs]

        # second parent
        p2_paths = shuffled_paths[num_pairs : 2 * num_pairs]
        p2_trans = shuffled_transponders[num_pairs : 2 * num_pairs]

        # remainder
        rem_paths = shuffled_paths[2 * num_pairs :]
        rem_trans = shuffled_transponders[2 * num_pairs :]

        c1_paths, c2_paths = self._cross_component(
            p1_paths, p2_paths, self.path_edge_bandwidth_usage_cross_proba
        )
        c1_trans, c2_trans = self._cross_component(
            p1_trans, p2_trans, self.path_transponder_assignment_cross_proba
        )

        crossed_paths = torch.cat((c1_paths, c2_paths, rem_paths), dim=0)
        crossed_trans = torch.cat((c1_trans, c2_trans, rem_trans), dim=0)

        # add elites back
        new_paths = torch.cat((elite_paths, crossed_paths), dim=0)
        new_transponders = torch.cat(
            (elite_transponders, crossed_trans), dim=0
        )

        return Population.masked(
            encrypted_neigh_matrix=population.encrypted_neigh_matrix,
            path_edge_bandwidth_usage=new_paths,
            path_transponder_assignment=new_transponders,
            neigh_matrix=neigh_matrix,
            encrypted_bandwidth=population.encrypted_bandwidth,
            regular_bandwidth=population.regular_bandwidth
        )
