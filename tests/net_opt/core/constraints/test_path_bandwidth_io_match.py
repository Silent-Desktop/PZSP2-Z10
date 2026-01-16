import pytest
import torch
from net_opt.core.constraints.path_bandwidth_io_match import (
    PathBandwidthIOMatch,
)


class TestPathBandwidthIOMatch:

    @pytest.fixture
    def constraint(self):
        return PathBandwidthIOMatch()

    @pytest.fixture
    def empty_inputs(self, sample_data):
        """
        Helper to create dummy inputs
        """
        return {
            "transponder_capacities": torch.zeros(sample_data["T"]),
            "demand": torch.zeros(sample_data["N"], sample_data["N"]),
        }

    def test_perfect_match(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: For every path (i->j), the flow leaving i exactly matches the flow entering j.
        This is a direct link without intermediate nodes.
        Expected: Similarity score should be 0.0 (perfect match).
        """
        P, N = sample_data["P"], sample_data["N"]

        usage = torch.zeros(P, N, N, N, N)

        # Path 0->1, edge 0 - 1
        usage[0, 0, 1, 0, 1] = 10.0

        pop = population_factory(bandwidth_usage=usage)
        results = constraint._check_all(pop, **empty_inputs)

        assert results[0, 0, 1] == 0.0

    def test_mismatch_flow(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Scenario: Flow leaving source is 10, but flow arriving at dest is 5.
        This implies loss or bad routing, resulting in a penalty score.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)

        # Path 0->2, edge 0 - 1
        usage[0, 0, 1, 0, 2] = 10.0

        # Path 0->2, edge 1 - 2
        usage[0, 1, 2, 0, 2] = 5.0  # Less outgoing than incoming

        pop = population_factory(bandwidth_usage=usage)
        results = constraint._check_all(pop, **empty_inputs)

        assert torch.isclose(results[0, 0, 2], torch.tensor(0.5))

    def test_einsum_aggregation_complex(
        self, constraint, population_factory, empty_inputs
    ):
        """
        Verify the einsum handles multiple output links correctly.
        Requires N=4, so we override the default dimensions.
        """
        P, N = 2, 4
        usage = torch.zeros(P, N, N, N, N)

        # Path 0->3, edge 0 - 1
        usage[0, 0, 1, 0, 3] = 10.0
        # Path 0->3, edge 0 - 2
        usage[0, 0, 2, 0, 3] = 20.0

        # Path 0->3, edge 1 - 3
        usage[0, 1, 3, 0, 3] = 15.0
        usage[0, 2, 3, 0, 3] = 15.0

        pop = population_factory(bandwidth_usage=usage)
        custom_inputs = {
            "transponder_capacities": torch.zeros(1),
            "demand": torch.zeros(N, N),
        }

        results = constraint._check_all(pop, **custom_inputs)

        assert results[0, 0, 3] == 0.0  # Perfect match: Out=30, In=30

    def test_check_aggregation(
        self, constraint, population_factory, empty_inputs, sample_data
    ):
        """
        Verify the main check() method correctly averages the scores across the population.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)

        # Pop 0: Perfect match everywhere (all zeros)
        # Pop 1: Massive mismatch. Path 0->1 has 10 outgoing but 0 incoming.
        usage[1, 0, 1, 0, 2] = 10.0

        pop = population_factory(bandwidth_usage=usage)

        scores = constraint.check(pop, **empty_inputs)

        assert scores.shape == (P,)

        assert scores[0] == 0.0  # Pop 0: perfect match

        expected_score_p1 = 1.0 / (
            N * N
        )  # Only one path has mismatch, rest are perfect

        # This encourages small changes because the penalty is diluted over all paths

        assert torch.isclose(scores[1], torch.tensor(expected_score_p1))
