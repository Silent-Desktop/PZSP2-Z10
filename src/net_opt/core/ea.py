from pydantic import BaseModel, Field, computed_field
from pydantic.config import ConfigDict
from torch.distributions import Distribution
import torch
from jaxtyping import Float, Bool, jaxtyped
from typing import Literal
from torch import Tensor
from beartype import beartype
import networkx as nx
import pprint as pp
from net_opt.core.population import Population
from net_opt.core.constraints.base_constraint import Constraint
from net_opt.core.termination_conditions.base_termination_condition import (
    TerminationCondition,
)
from net_opt.core.mutations.base_mutation import Mutation
from net_opt.core.selections.base_selection import Selection
from net_opt.utils.visualisation import visualize_population_individual


class EA(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    population_size: int
    elite_size: int

    path_edge_bandwidth_usage_init_distribution: Distribution
    path_transponder_assignment_init_distribution: Distribution

    global_constraint_weight: float = 10.0**5
    constraints: list[Constraint] = Field(default_factory=lambda _: [])

    termination_conditions: list[TerminationCondition] = Field(
        default_factory=lambda _: []
    )
    selection_method: Selection
    mutation_methods: list[Mutation] = Field(default_factory=lambda _: [])

    show_vizualisation_every_n_iter: int = 10000
    device: torch.device

    @computed_field
    @property
    @jaxtyped(typechecker=beartype)
    def constraint_weights_tensor(self) -> Float[Tensor, "C"]:
        constraint_weights = [c.weight for c in self.constraints]
        return torch.tensor(constraint_weights, dtype=torch.float32)

    def _get_encrypted_MST(
        self, neigh_matrix: Bool[Tensor, "N N"]
    ) -> Bool[Tensor, "N N"]:
        OG = nx.from_numpy_array(neigh_matrix.cpu().numpy())
        betweenness = nx.edge_betweenness_centrality(OG, normalized=False)
        nx.set_edge_attributes(OG, betweenness, "OG")
        N = OG.number_of_nodes() - 1
        G = nx.Graph(OG)
        T = nx.Graph()
        while T.number_of_edges() < N:
            betweenness = nx.edge_betweenness_centrality(G, normalized=False)
            nx.set_edge_attributes(G, betweenness, "temp")
            edge_queue = sorted(
                G.edges.data(),
                key=lambda edge: (edge[2]["OG"], edge[2]["temp"]),
            )
            for u, v, _ in edge_queue:
                try:
                    temp = nx.Graph(T)
                    temp.add_edge(u, v)
                    nx.find_cycle(temp, u)
                except nx.NetworkXNoCycle:
                    G.remove_edge(u, v)
                    T.add_edge(u, v)
                    break
        return (
            torch.from_numpy(nx.to_numpy_array(T, dtype=bool))
            .to(self.device)
            .triu_(diagonal=1)
        )

    def _sample_init_population(
        self,
        neigh_matrix: Bool[Tensor, "N N"],
        T: int,
        encrypted_bandwidth: int,
        regular_bandwidth: int,
    ) -> Population:
        encrypted_neigh_matrix = self._get_encrypted_MST(neigh_matrix)
        N = neigh_matrix.size(0)
        path_edge_bandwidth_usage_size = torch.Size(
            [self.population_size, N, N, N, N]
        )
        path_edge_bandwidth_usage = (
            self.path_edge_bandwidth_usage_init_distribution.sample(
                path_edge_bandwidth_usage_size
            ).float()
        )  # (P, N, N, N, N)
        path_transponder_assignment_size = torch.Size(
            [self.population_size, T, N, N]
        )
        path_transponder_assignment = (
            self.path_transponder_assignment_init_distribution.sample(
                path_transponder_assignment_size
            ).float()
        )  # (P, T, N, N)
        return Population.masked(
            encrypted_neigh_matrix=encrypted_neigh_matrix,
            path_edge_bandwidth_usage=path_edge_bandwidth_usage,
            path_transponder_assignment=path_transponder_assignment,
            neigh_matrix=neigh_matrix,
            encrypted_bandwidth=encrypted_bandwidth,
            regular_bandwidth=regular_bandwidth,
        )

    @jaxtyped(typechecker=beartype)
    def _check_constraints(self) -> Float[Tensor, "C P"]:
        if not self.constraints:
            P = self._population.path_edge_bandwidth_usage.size(0)
            return torch.empty(0, P)
        all_scores = [
            c.check(
                self._population, self._transponder_capacities, self._demand
            )
            for c in self.constraints
        ]

        return torch.stack(all_scores, dim=0)

    @jaxtyped(typechecker=beartype)
    def _calculate_total_transponder_cost(self) -> Float[Tensor, "P"]:
        path_costs = (
            self._population.path_transponder_assignment.permute(0, 2, 3, 1)
            @ self._transponder_costs
        )  # (P, N, N, T) @ (T) -> (P, N, N)
        return path_costs.sum(dim=(1, 2))  # (P, N, N).sum(dim=(1, 2)) -> (P,)

    @jaxtyped(typechecker=beartype)
    def product_penalty(self) -> Float[Tensor, "P"]:
        constraint_product = torch.prod(
            1 - self._constraint_scores, dim=0
        )  # (P)
        return (
            self._total_transponder_cost / constraint_product
            + self.global_constraint_weight * (1 - constraint_product)
        )

    @jaxtyped(typechecker=beartype)
    def sum_penalty(self) -> Float[Tensor, "P"]:
        w_constraint_total = (
            self.constraint_weights_tensor @ self._constraint_scores
        )  # (P)
        return self._total_transponder_cost + w_constraint_total * (
            self._total_transponder_cost + self.global_constraint_weight
        )

    @jaxtyped(typechecker=beartype)
    def run(
        self,
        encrypted_bandwidth: int,
        regular_bandwidth: int,
        neigh_matrix: Bool[Tensor, "N N"],
        demand: Float[Tensor, "N N"],
        transponder_costs: Float[Tensor, "T"],
        transponder_capacities: Float[Tensor, "T"],
        penalty_method: Literal["prod", "sum"] = "prod",
    ):
        self._run_init(
            encrypted_bandwidth,
            regular_bandwidth,
            neigh_matrix,
            demand,
            transponder_costs,
            transponder_capacities,
            penalty_method,
        )
        visualize_population_individual(
            self._population, transponder_capacities, 0, self._iteration_n
        )
        while all(
            [
                cond.check(self._lowest_penalty, self._iteration_n)
                for cond in self.termination_conditions
            ]
        ):
            try:
                self._precalc()

                if (
                    self._iteration_n % self.show_vizualisation_every_n_iter
                    == 0
                ):
                    visualize_population_individual(
                        self._population,
                        self._transponder_capacities,
                        0,
                        self._iteration_n,
                    )
                dict_constraint_name_scores = {
                    self.constraints[
                        i
                    ].readable_name: self._lowest_penalty_constraint_scores[i]
                    for i in range(len(self._constraint_scores))
                }
                print("-" * 40)
                print(f"it: {self._iteration_n}")
                print(f"lowest_penalty: {self._lowest_penalty}")
                print("lowest_penalty_constraint_scores:")
                pp.pprint(dict_constraint_name_scores)
                print(
                    f"lowest_transponder_cost: {self._lowest_transponder_cost}"
                )

                self._postcalc()
            except KeyboardInterrupt:
                print("===INTERRUPTED===")
                visualize_population_individual(
                    self._population,
                    self._transponder_capacities,
                    0,
                    self._iteration_n,
                )
                return

    def _run_init(
        self,
        encrypted_bandwidth: int,
        regular_bandwidth: int,
        neigh_matrix: Bool[Tensor, "N N"],
        demand: Float[Tensor, "N N"],
        transponder_costs: Float[Tensor, "T"],
        transponder_capacities: Float[Tensor, "T"],
        penalty_method: Literal["prod", "sum"] = "prod",
    ):
        if self.elite_size >= self.population_size:
            raise ValueError("Elite should be smaller than the population!")
        self._penalty = (
            self.sum_penalty
            if penalty_method == "sum"
            else self.product_penalty
        )
        self._demand = demand
        self._transponder_costs = transponder_costs
        self._transponder_capacities = transponder_capacities
        T = transponder_capacities.size(0)
        self._population = self._sample_init_population(
            neigh_matrix, T, encrypted_bandwidth, regular_bandwidth
        )
        self._lowest_penalty = float("inf")
        self._iteration_n = 0
        self._lowest_penalty_constraint_scores: list[float] = [
            1.0 for _ in self.constraints
        ]
        self._lowest_transponder_cost = float("inf")

    def _precalc(self):
        self._iteration_n += 1
        self._constraint_scores = self._check_constraints()  # (C, P)
        self._total_transponder_cost = (
            self._calculate_total_transponder_cost()
        )  # (P)
        self._penalties = self._penalty()
        # TODO: no need to look through penalties twice. Refactor
        self._curr_lowest_penalty = self._penalties.min().item()

    def _postcalc(self):
        if self._curr_lowest_penalty < self._lowest_penalty:
            curr_lowest_penalty_idx = self._penalties.argmin()
            self._lowest_penalty_constraint_scores = self._constraint_scores[:, curr_lowest_penalty_idx].tolist()  # type: ignore
            self._lowest_penalty = self._curr_lowest_penalty
            self._lowest_transponder_cost = self._total_transponder_cost[
                curr_lowest_penalty_idx
            ].item()
        sorted_indices = torch.argsort(self._penalties)
        sorted_penalties = self._penalties[sorted_indices]
        sorted_population = Population(
            encrypted_neigh_matrix=self._population.encrypted_neigh_matrix,
            path_edge_bandwidth_usage=self._population.path_edge_bandwidth_usage[
                sorted_indices
            ],
            path_transponder_assignment=self._population.path_transponder_assignment[
                sorted_indices
            ],
            encrypted_bandwidth=self._population.encrypted_bandwidth,
            regular_bandwidth=self._population.regular_bandwidth,
            neigh_matrix=self._population.neigh_matrix,
        )
        next_generation = self.selection_method.get_next_generation(
            sorted_population, sorted_penalties, self.elite_size
        )
        for mutation in self.mutation_methods:
            next_generation = mutation.mutate(
                next_generation, self.elite_size, self._population.neigh_matrix
            )
        self._population = next_generation


class FastEA(EA):
    @jaxtyped(typechecker=beartype)
    def run(
        self,
        neigh_matrix: Bool[Tensor, "N N"],
        demand: Float[Tensor, "N N"],
        transponder_costs: Float[Tensor, "T"],
        transponder_capacities: Float[Tensor, "T"],
    ):
        self._run_init(
            neigh_matrix, demand, transponder_costs, transponder_capacities
        )
        while all(
            [
                cond.check(self._lowest_penalty, self._iteration_n)
                for cond in self.termination_conditions
            ]
        ):
            self._precalc()
            self._postcalc()
