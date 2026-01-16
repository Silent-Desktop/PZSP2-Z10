import pytest
import torch
from unittest.mock import patch
from net_opt.core.selections.tournament_selection import TournamentSelection


class TestTournamentSelection:

    @pytest.fixture
    def selector(self):
        return TournamentSelection(k=2)

    @pytest.fixture
    def tagged_population(self, population_factory, sample_data):
        """
        Creates a population where each individual is uniquely identifiable.
        We tag on usage[i, 0, 1, 0, 1] = i.
        We avoid [0, 0, 0, 0, 0] because Population.masked zeroes out self-loops.
        """
        P, N, T = 4, sample_data["N"], sample_data["T"]

        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, T, N, N)

        # Tag individuals with an ID
        for i in range(P):
            usage[i, 0, 1, 0, 1] = float(i)
            transponders[i, 0, 1, 0] = float(i)

        return population_factory(
            bandwidth_usage=usage,
            path_transponder_assignment=transponders,
            P=P,
        )

    def test_initialization(self):
        ts = TournamentSelection(k=3)
        assert ts.k == 3

    def test_elitism_preservation(self, selector, tagged_population):
        """
        Verify that the top 'elite_size' individuals are copied directly.
        """
        penalties = torch.zeros(4)
        elite_size = 1

        new_pop = selector.get_next_generation(
            tagged_population, penalties, elite_size
        )

        # Check if the first individual is exactly individual 0 (ID 0.0)
        assert new_pop.path_edge_bandwidth_usage[0, 0, 1, 0, 1] == 0.0

    @patch("net_opt.core.selections.tournament_selection.torch.randint")
    def test_tournament_logic_deterministic(
        self, mock_randint, selector, tagged_population
    ):
        """
        Scenario: P=4. Elite=0. Select 4 individuals.
        Penalties: [100, 10, 50, 0] -> IDs: [0, 1, 2, 3]

        Tournaments:
        [0, 1] -> 1 wins
        [2, 3] -> 3 wins
        [0, 2] -> 2 wins
        [1, 3] -> 3 wins

        Expected: [1, 3, 2, 3]
        """
        penalties = torch.tensor([100.0, 10.0, 50.0, 0.0])
        elite_size = 0
        P = 4

        mock_indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
        mock_randint.return_value = mock_indices

        new_pop = selector.get_next_generation(
            tagged_population, penalties, elite_size
        )

        mock_randint.assert_called_with(low=0, high=P, size=(4, 2))

        result_ids = new_pop.path_edge_bandwidth_usage[:, 0, 1, 0, 1]
        expected_ids = torch.tensor([1.0, 3.0, 2.0, 3.0])

        assert torch.equal(result_ids, expected_ids)

    def test_output_tensor_shapes(self, selector, tagged_population):
        """
        Ensure the population doesnt change size or shape after selection.
        """
        penalties = torch.zeros(4)
        new_pop = selector.get_next_generation(
            tagged_population, penalties, elite_size=1
        )

        assert (
            new_pop.path_edge_bandwidth_usage.shape
            == tagged_population.path_edge_bandwidth_usage.shape
        )
        assert (
            new_pop.path_transponder_assignment.shape
            == tagged_population.path_transponder_assignment.shape
        )
        assert (
            new_pop.encrypted_neigh_matrix.shape
            == tagged_population.encrypted_neigh_matrix.shape
        )
