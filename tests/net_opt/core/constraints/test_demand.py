import pytest
import torch
from net_opt.core.constraints.demand import Demand


class TestDemand:

    @pytest.fixture
    def constraint(self):
        return Demand()

    @pytest.fixture
    def transponder_caps(self, sample_data):
        """
        Sample capacities for the T=4 transponder types.
        """
        return torch.tensor([10.0, 40.0, 100.0, 400.0])

    def test_exact_match(
        self, constraint, population_factory, transponder_caps, sample_data
    ):
        """
        Scenario: Demand is exactly met by the transponder capacity.
        Demand: 100.
        Capacity: 1x 100G transponder (Idx 2).
        Expected: Score 0.0
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]

        demand = torch.zeros(N, N)
        demand[0, 1] = 100.0

        transponders = torch.zeros(P, T, N, N)
        # Pop 0, Type 2 (100G), Path 0->1 -> 1 unit
        transponders[0, 2, 0, 1] = 1.0

        pop = population_factory(path_transponder_assignment=transponders)

        results = constraint._check_all(pop, transponder_caps, demand)

        # Diff = 0
        assert results[0, 0, 1] == 0.0

    def test_over_provisioned(
        self, constraint, population_factory, transponder_caps, sample_data
    ):
        """
        Scenario: Capacity exceeds Demand.
        Demand: 50.
        Capacity: 1x 100G transponder.
        Expected: Score 0.0
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]

        demand = torch.zeros(N, N)
        demand[0, 1] = 50.0

        transponders = torch.zeros(P, T, N, N)
        # Pop 0, Path 0->1, Type 2 (100G)
        transponders[0, 2, 0, 1] = 1.0

        pop = population_factory(path_transponder_assignment=transponders)
        results = constraint._check_all(pop, transponder_caps, demand)

        # Diff = ReLU(50 - 100) = ReLU(-50) = 0
        assert results[0, 0, 1] == 0.0

    def test_under_provisioned(
        self, constraint, population_factory, transponder_caps, sample_data
    ):
        """
        Scenario: Capacity is less than Demand.
        Demand: 100.
        Capacity: 1x 40G transponder (Idx 1).
        Expected: Penalty score.
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]

        demand = torch.zeros(N, N)
        demand[0, 1] = 100.0

        transponders = torch.zeros(P, T, N, N)
        # Pop 0, Path 0->1, Type 1 (40G)
        transponders[0, 1, 0, 1] = 1.0

        pop = population_factory(path_transponder_assignment=transponders)
        results = constraint._check_all(pop, transponder_caps, demand)

        # Diff = ReLU(100 - 40) = 60
        # Goal = Max(100, 40) = 100
        # Score = 60 / 100 = 0.6
        assert torch.isclose(results[0, 0, 1], torch.tensor(0.6))

    def test_zero_demand_zero_capacity(
        self, constraint, population_factory, transponder_caps, sample_data
    ):
        """
        Scenario: No demand for a path, and no transponders assigned.
        Expected: 0.0 (nan_to_num handles 0/0).
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]

        demand = torch.zeros(N, N)
        transponders = torch.zeros(P, T, N, N)

        pop = population_factory(path_transponder_assignment=transponders)
        results = constraint._check_all(pop, transponder_caps, demand)

        assert results[0, 0, 1] == 0.0

    def test_check_aggregation_sums(
        self, constraint, population_factory, transponder_caps, sample_data
    ):
        """
        Verify the aggregation logic in check().
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]

        demand = torch.zeros(N, N)
        demand[0, 1] = 100.0
        demand[0, 2] = 50.0

        transponders = torch.zeros(P, T, N, N)

        # Pop 0: perfect match
        transponders[0, 2, 0, 1] = 1.0
        transponders[0, 1, 0, 2] = 1.0
        transponders[0, 0, 0, 2] = 1.0

        # Pop 1: under-provisioned on both paths
        transponders[1, 1, 0, 2] = 1.0
        transponders[1, 0, 0, 2] = 1.0

        pop = population_factory(path_transponder_assignment=transponders)

        scores = constraint.check(pop, transponder_caps, demand)

        # Pop 0 should be perfect
        assert scores[0] == 0.0

        # Pop 1: under-provisioned on both paths
        # Path 0->1: Demand 100, Cap 0.   Diff = 100. Goal = 100.
        # Path 0->2: Demand 50,  Cap 50.  Diff = 0.   Goal = 50.
        # Other paths: Demand 0, Cap 0.   Diff = 0.   Goal = 0.

        # Sum of Diffs = 100
        # Sum of Goals = 100 + 50 = 150
        # Score = 100 / 150

        expected_score = 100.0 / 150.0
        assert torch.isclose(scores[1], torch.tensor(expected_score))
