import pytest
import torch
from unittest.mock import patch, MagicMock
from net_opt.core.mutations.uniform_crossover import UniformCrossover

class TestUniformCrossover:

    @pytest.fixture
    def crossover(self):
        return UniformCrossover(
            path_edge_bandwidth_usage_cross_proba=0.5,
            path_transponder_assignment_cross_proba=0.5
        )

    @pytest.fixture
    def labeled_population(self, population_factory, sample_data):
        """
        Creates a population where each individual is uniquely identifiable.
        Uses valid edge indices (Path 0->1, Edge 0->1) to avoid masking issues.
        Ind 0 = 0.0, Ind 1 = 1.0, etc.
        """
        P, N, T = 4, sample_data["N"], sample_data["T"]
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, N, N, T)

        for i in range(P):
            val = float(i)
            usage[i] = val
            transponders[i] = val
        
        return population_factory(
            bandwidth_usage=usage, 
            path_transponder_assignment=transponders,
            P=P
        )

    def test_initialization(self):
        uc = UniformCrossover(
            path_edge_bandwidth_usage_cross_proba=0.2,
            path_transponder_assignment_cross_proba=0.8
        )
        assert uc.path_edge_bandwidth_usage_cross_proba == 0.2
        assert uc.path_transponder_assignment_cross_proba == 0.8

    def test_elitism_preservation(self, crossover, labeled_population):
        """
        Verify that the elite_size individuals are returned exactly as they were.
        """
        
        elite_size = 1
        original_elite = labeled_population.path_edge_bandwidth_usage[0].clone()
        crossover.path_edge_bandwidth_usage_cross_proba = 1.0
        
        new_pop = crossover.mutate(labeled_population, elite_size)
        
        assert torch.equal(new_pop.path_edge_bandwidth_usage[0], original_elite)

    @patch("net_opt.core.mutations.uniform_crossover.torch.randperm")
    @patch("net_opt.core.mutations.uniform_crossover.Bernoulli")
    def test_crossover_mechanics(self, MockBernoulli, MockRandperm, crossover, labeled_population):
        """
        Verify that children correctly mix parts from parents.
        Implementation splits population (P=4) into halves: P1=[0,1], P2=[2,3].
        
        Pair 1: Ind 0 (0.0) & Ind 2 (2.0).
        Pair 2: Ind 1 (1.0) & Ind 3 (3.0).
        """
        P = 4
        elite_size = 0
        
        # Pairing (Identity)
        MockRandperm.return_value = torch.arange(P)
        
        # Crossover Mask: 
        # Index 1 (Link 0->1) is True (Take P1)
        # Index 2 (Link 0->2) is False (Take P2)
        mock_dist = MagicMock()
        MockBernoulli.return_value = mock_dist
        
        def side_effect_sample(size):
            mask = torch.zeros(size)
            mask[..., 1] = 1.0 
            return mask
            
        mock_dist.sample.side_effect = side_effect_sample
        
        new_pop = crossover.mutate(labeled_population, elite_size)
        
        # Pair 1 (Parents 0.0 and 2.0)
        assert new_pop.path_edge_bandwidth_usage[0, 0, 1, 0, 1] == 0.0
        assert new_pop.path_edge_bandwidth_usage[0, 0, 1, 0, 2] == 2.0
        
        # Pair 2 (Parents 1.0 and 3.0)
        assert new_pop.path_edge_bandwidth_usage[1, 0, 1, 0, 1] == 1.0
        assert new_pop.path_edge_bandwidth_usage[1, 0, 1, 0, 2] == 3.0