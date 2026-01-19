import pytest
import torch
from net_opt.core.constraints.path_transponder_bandwidth_match import (
    PathTransponderBandwidthMatch,
)


class TestPathTransponderBandwidthMatch:

    @pytest.fixture
    def constraint(self):
        return PathTransponderBandwidthMatch()

    @pytest.fixture
    def empty_inputs(self, sample_data):
        """
        Helper to create dummy inputs.
        """
        return {
            "transponder_capacities": torch.zeros(sample_data["T"]),
            "demand": torch.zeros(sample_data["N"], sample_data["N"]),
        }

    def test_perfect_match(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Total bandwidth leaving source equals the sum of assigned transponders.
        Expected: Similarity score should be 0.0.
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Path 0->1: 100 Bandwidth vs 100 Transponders (split 60+40 across types)
        usage[0, 0, 1, 0, 1] = 100.0
        transponders[0, 0, 0, 1] = 60.0
        transponders[0, 1, 0, 1] = 40.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        results = constraint._check_all(pop, **empty_inputs)

        assert results[0, 0, 1] == 0.0

    def test_mismatch_under_provisioned(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Bandwidth usage (100) is higher than assigned transponders (50).
        Expected: Penalty score reflecting under-provisioning.
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Path 0->2: 100 Bandwidth vs 50 Transponders -> 0.5 error
        usage[0, 0, 1, 0, 2] = 100.0
        transponders[0, 0, 0, 2] = 50.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        results = constraint._check_all(pop, **empty_inputs)

        assert torch.isclose(results[0, 0, 2], torch.tensor(0.5))

    def test_mismatch_over_provisioned(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Assigned transponders (200) exceed bandwidth usage (100).
        Expected: Penalty score reflecting over-provisioning (waste).
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Path 0->2: 100 Bandwidth vs 200 Transponders -> 0.5 error
        usage[0, 0, 1, 0, 2] = 100.0
        transponders[0, 2, 0, 2] = 200.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        results = constraint._check_all(pop, **empty_inputs)

        assert torch.isclose(results[0, 0, 2], torch.tensor(0.5))

    def test_check_aggregation(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Verify the main check() method correctly averages the scores across the population.
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Pop 1: Massive mismatch on ONE path (10 Bandwidth, 0 Transponders)
        usage[1, 0, 1, 0, 1] = 10.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        scores = constraint.check(pop, **empty_inputs)

        assert scores.shape == (P,)
        assert scores[0] == 0.0

        # Only one path has error, diluted by N*N total paths
        expected_score_p1 = 1.0 / (N * N)
        assert torch.isclose(scores[1], torch.tensor(expected_score_p1))
