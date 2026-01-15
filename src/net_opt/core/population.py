from typing import NamedTuple
from jaxtyping import Bool, Float, jaxtyped
from pydantic import computed_field
from torch import Tensor
from beartype import beartype


class Population(NamedTuple):
    encrypted_neigh_matrix: Bool[Tensor, "N N"]
    path_edge_bandwidth_usage: Float[Tensor, "P N N N N"]
    path_transponder_assignment: Float[Tensor, "P T N N"]
    encrypted_bandwidth: int
    regular_bandwidth: int

    @classmethod
    @jaxtyped(typechecker=beartype)
    def masked(
        cls,
        encrypted_neigh_matrix: Bool[Tensor, "N N"],
        path_edge_bandwidth_usage: Float[Tensor, "P N N N N"],
        path_transponder_assignment: Float[Tensor, "P T N N"],
        encrypted_bandwidth: int,
        regular_bandwidth: int,
        neigh_matrix: Bool[Tensor, "N N"],
    ):
        # mask undirected node pairs
        path_edge_bandwidth_usage.triu_(diagonal=1)
        # mask nonexistent edges
        path_edge_bandwidth_usage[:, (~neigh_matrix), :, :] = 0.0
        # mask undirected node pairs
        path_transponder_assignment.triu_(diagonal=1)
        return cls(
            encrypted_neigh_matrix,
            path_edge_bandwidth_usage,
            path_transponder_assignment,
            encrypted_bandwidth,
            regular_bandwidth
        )

    @computed_field
    @property
    @jaxtyped(typechecker=beartype)
    def edge_size_limits(self) -> Float[Tensor, "N N"]:
        return (
            self.encrypted_neigh_matrix.float() * self.encrypted_bandwidth
            + (~self.encrypted_neigh_matrix).float() * self.regular_bandwidth
        )
