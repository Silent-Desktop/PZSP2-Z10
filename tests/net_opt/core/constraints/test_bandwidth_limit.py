import pytest
import torch
from unittest.mock import patch, PropertyMock
from net_opt.core.constraints.bandwidth_limit import BandwidthLimit

class TestBandwidthLimit:

    @pytest.fixture
    def constraint(self):
        return BandwidthLimit()

    @pytest.fixture
    def empty_inputs(self, sample_data):
        return {
            "transponder_capacities": torch.zeros(sample_data["T"]),
            "demand": torch.zeros(sample_data["N"], sample_data["N"])
        }

    def test_under_limit(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Usage (50) is well below the limit.
        Expected: Score 0.0.
        """
        P, N = sample_data["P"], sample_data["N"]
        
        # Setup: Usage on link 0->1 is 50.0
        usage = torch.zeros(P, N, N, N, N)
        usage[0, 0, 1, 0, 1] = 50.0
        
        pop = population_factory(bandwidth_usage=usage)

        # Mock edge_size_limits to be 100.0 everywhere
        with patch("net_opt.core.population.Population.edge_size_limits", new_callable=PropertyMock) as mock_limits:
            mock_limits.return_value = torch.ones(N, N) * 100.0
            
            results = constraint._check_all(pop, **empty_inputs)
            
            # Diff = ReLU(50 - 100) = 0
            assert results[0, 0, 1] == 0.0

    def test_over_limit(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Usage (150) exceeds the limit (100).
        Expected: Penalty score.
        """
        P, N = sample_data["P"], sample_data["N"]
        
        usage = torch.zeros(P, N, N, N, N)
        usage[0, 0, 1, 0, 1] = 150.0
        
        pop = population_factory(bandwidth_usage=usage)

        with patch("net_opt.core.population.Population.edge_size_limits", new_callable=PropertyMock) as mock_limits:
            mock_limits.return_value = torch.ones(N, N) * 100.0
            
            results = constraint._check_all(pop, **empty_inputs)
            
            # Usage = 150. Limit = 100.
            # Diff = 50. Max = 150.
            # Score = 50 / 150 = 1/3
            assert torch.isclose(results[0, 0, 1], torch.tensor(1.0/3.0))

    def test_bidirectional_summation(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Traffic flows both 0->1 and 1->0.
        The constraint sums them onto the upper triangle (0->1).
        Usage 0->1: 60.
        Usage 1->0: 50.
        Total on Link (0,1): 110.
        Limit: 100.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Path A uses Link 0->1 (60)
        usage[0, 0, 1, 0, 1] = 60.0
        # Path B uses Link 1->0 (50)
        usage[0, 0, 1, 1, 0] = 50.0
        
        pop = population_factory(bandwidth_usage=usage)

        with patch("net_opt.core.population.Population.edge_size_limits", new_callable=PropertyMock) as mock_limits:
            mock_limits.return_value = torch.ones(N, N) * 100.0
            
            results = constraint._check_all(pop, **empty_inputs)
            
            # Total Usage on physical edge (0,1) = 60 + 50 = 110.
            # Limit = 100.
            # Diff = 10. Max = 110.
            # Score = 10/110.
            assert torch.isclose(results[0, 0, 1], torch.tensor(10.0/110.0))

    def test_ignore_self_loops_and_lower_triangle(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Verify that the output only contains values in the triu (diagonal=1),
        ignoring self-loops (diagonal=0) and the lower triangle.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        # Add usage to a self-loop 0->0
        usage[0, 0, 0, 0, 0] = 1000.0
        
        pop = population_factory(bandwidth_usage=usage)
        
        with patch("net_opt.core.population.Population.edge_size_limits", new_callable=PropertyMock) as mock_limits:
            mock_limits.return_value = torch.ones(N, N) * 100.0
            
            results = constraint._check_all(pop, **empty_inputs)
            
            # Diagonal (0,0) should be 0 because triu_(diagonal=1) excludes it
            assert results[0, 0, 0] == 0.0
            
            # Lower triangle (1,0) should be 0
            assert results[0, 1, 0] == 0.0

    def test_check_aggregation(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Verify aggregation logic in check().
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Pop 1:
        # Link 0->1: Usage 150 (Limit 100) -> Deficit 50.
        # Link 0->2: Usage 100 (Limit 100) -> Deficit 0.
        usage[1, 0, 1, 0, 1] = 150.0
        usage[1, 0, 2, 0, 2] = 100.0
        
        pop = population_factory(bandwidth_usage=usage)

        with patch("net_opt.core.population.Population.edge_size_limits", new_callable=PropertyMock) as mock_limits:
            mock_limits.return_value = torch.ones(N, N) * 100.0
            
            scores = constraint.check(pop, **empty_inputs)
            
            # Pop 0: Score 0.
            assert scores[0] == 0.0
            
            # Pop 1:
            # Sum of Diffs = 50.
            # Sum of Goals:
            # Element (0,1) is max(150, 100) = 150.
            # All other elements are max(..., 100) = 100.
            # Total Goal = 150 + (N*N - 1) * 100 = 150 + 800 = 950.
            
            expected_score = 50.0 / 950.0
            assert torch.isclose(scores[1], torch.tensor(expected_score))