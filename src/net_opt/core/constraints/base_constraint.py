from abc import ABC, abstractmethod

from pydantic import BaseModel
from jaxtyping import Float, jaxtyped
from typing import Callable
from torch import Tensor
import torch
from beartype import beartype

from net_opt.core.population import Population


class Constraint(ABC, BaseModel):
    weight: float = 1.0
    readable_name: str
    @abstractmethod
    def check(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        """Calculate compliance score for each individual [0,1]"""
        ...
    @abstractmethod
    def _check_all(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        """Calculate compliance score for each element [0,1]"""
        ...
    @jaxtyped(typechecker=beartype)
    def _mean_similarity_scores(self, scores: Float[Tensor, "P ..."]) -> Float[Tensor, "P"]:
        return torch.mean(scores.flatten(start_dim=1), dim=1)

    @jaxtyped(typechecker=beartype)
    def _calculate_similarity_scores_all(self, a: Float[Tensor, "P ..."], b: Float[Tensor, "P ..."], diff_transform: Callable[[Tensor], Tensor]= torch.abs) -> Float[Tensor, "P ..."]:
        diff = diff_transform(a - b) 
        goal = torch.max(a, b)
        ratio = diff / goal
        # this one has no mean for DependentDemand
        return ratio.nan_to_num_()
    
    @jaxtyped(typechecker=beartype)
    def _calculate_similarity_scores_from_sums(self, a: Float[Tensor, "P ..."], b: Float[Tensor, "P ..."], diff_transform: Callable[[Tensor], Tensor]= torch.abs) -> Float[Tensor, "P"]:
        # this one is useful for hard constraints that use relu (demand, bandwidth limit)
        diff = diff_transform(a - b) 
        goal = torch.max(a, b)
        per_population_diff = diff.flatten(start_dim=1).sum(dim=1)
        per_population_goal = goal.flatten(start_dim=1).sum(dim=1)
        ratio = per_population_diff / per_population_goal
        # For zero division
        return ratio.nan_to_num_()