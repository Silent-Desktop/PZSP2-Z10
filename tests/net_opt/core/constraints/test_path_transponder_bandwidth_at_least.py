import pytest
import torch
from net_opt.core.constraints.path_transponder_bandwidth_weak import (
    PathTransponderBandwidthWeak,
)


class TestPathTransponderBandwidthWeak:

    @pytest.fixture
    def constraint(self):
        return PathTransponderBandwidthWeak()

    @pytest.fixture
    def empty_inputs(self, sample_data):
        """
        Helper to create dummy inputs.
        """
        return {
            "transponder_capacities": torch.zeros(sample_data["T"]),
            "demand": torch.zeros(sample_data["N"], sample_data["N"]),
        }

    def test_compliant_bandwidth_exceeds_transponders(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Bandwidth usage (100) is greater than Transponders (50).
        Logic: relu(Transponders - Bandwidth) = relu(50 - 100) = 0.
        Expected: Score 0.0 (Compliant).
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Path 0->1
        usage[0, 0, 1, 0, 1] = 100.0
        transponders[0, 0, 1, 0] = 50.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        results = constraint._check_all(pop, **empty_inputs)

        assert results[0, 0, 1] == 0.0

    def test_compliant_exact_match(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Bandwidth usage (100) exactly equals Transponders (100).
        Logic: relu(100 - 100) = 0.
        Expected: Score 0.0 (Compliant).
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Path 0->1
        usage[0, 0, 1, 0, 1] = 100.0
        transponders[0, 0, 0, 1] = 100.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        results = constraint._check_all(pop, **empty_inputs)

        assert results[0, 0, 1] == 0.0

    def test_violation_transponders_exceed_bandwidth(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Transponders (100) exceed Bandwidth usage (50).
        Logic: relu(Transponders - Bandwidth) = relu(100 - 50) = 50.
        Max(100, 50) = 100.
        Expected: Score 50/100 = 0.5.
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Path 0->1: Over-provisioned
        usage[0, 0, 1, 0, 1] = 50.0
        transponders[0, 0, 0, 1] = 100.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        results = constraint._check_all(pop, **empty_inputs)

        assert torch.isclose(results[0, 0, 1], torch.tensor(0.5))

    def test_check_aggregation(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Verify the main check() method correctly averages the scores across the population.
        """
        P, N, T = sample_data["P"], sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Pop 1: Violation on ONE path (Path 0->1)
        # Transponders (10) > Usage (0). Score = 10/10 = 1.0.
        transponders[1, 0, 0, 1] = 10.0
        usage[1, 0, 1, 0, 1] = 0.0

        pop = population_factory(
            bandwidth_usage=usage, path_transponder_assignment=transponders
        )
        scores = constraint.check(pop, **empty_inputs)

        assert scores.shape == (P,)
        assert scores[0] == 0.0

        expected_score_p1 = 1.0 / (N * N)  # Diluted penalty
        assert torch.isclose(scores[1], torch.tensor(expected_score_p1))
