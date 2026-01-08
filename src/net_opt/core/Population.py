# ruff: noqa: F722
# ruff: noqa: F821
# Above from: https://docs.kidger.site/jaxtyping/faq/

from typing import NamedTuple
from jaxtyping import Bool, Float, jaxtyped
from pydantic import computed_field
from torch import Tensor
import torch
from beartype import beartype

from net_opt.utiils.torch_utils import multi_triu

class Population(NamedTuple):
    encrypted_neigh_matrix: Bool[Tensor, "N N"]
    path_edge_bandwidth_usage: Float[Tensor, "P N N N N"]
    path_transponder_assignment: Float[Tensor, "P N N T"]
    regular_bandwidth: int = 96
    encrypted_bandwidth: int = 30

    @classmethod
    @jaxtyped(typechecker=beartype)
    def masked(cls, 
               encrypted_neigh_matrix: Bool[Tensor, "N N"],
               path_edge_bandwidth_usage: Float[Tensor, "P N N N N"],
               path_transponder_assignment: Float[Tensor, "P N N T"]
               ):
        idx = torch.arange(0, path_edge_bandwidth_usage.size(1))
        path_edge_bandwidth_usage[:, :, :, idx, idx] = 0.0
        new_transponders_masked = multi_triu(path_transponder_assignment, dim_pairs=[(1,2)]) # (P, *N, *N, T)
        return cls(encrypted_neigh_matrix, path_edge_bandwidth_usage, new_transponders_masked)
    
    @computed_field
    @property
    @jaxtyped(typechecker=beartype)
    def edge_size_limits(self) -> Float[Tensor, "N N"]:
        return (self.encrypted_neigh_matrix.float() * self.encrypted_bandwidth + (~self.encrypted_neigh_matrix).float() * self.regular_bandwidth)
