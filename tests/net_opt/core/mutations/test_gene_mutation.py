import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from net_opt.core.mutations.gene_mutation import GeneMutation

class TestGeneMutation:

    @pytest.fixture
    def mutator(self):
        return GeneMutation(
            path_edge_bandwidth_usage_mut_proba=0.5,
            path_transponder_assignment_mut_proba=0.5
        )

    @pytest.fixture
    def tagged_population(self, population_factory, sample_data):
        """
        Creates a population where each individual is uniquely identifiable.
        Uses valid edge indices (Path 0->1, Edge 0->1) to avoid masking issues.
        Values start at 100.0 to ensure Poisson variance.
        """
        P, N, T = 4, sample_data["N"], sample_data["T"]
        
        usage = torch.zeros(P, N, N, N, N)
        transponders = torch.zeros(P, N, N, T)

        for i in range(P):
            val = float(i + 100)
            usage[i, 0, 1, 0, 1] = val
            transponders[i, 0, 1, 0] = val
        
        return population_factory(
            bandwidth_usage=usage, 
            path_transponder_assignment=transponders,
            P=P
        )

    def test_initialization(self):
        gm = GeneMutation(
            path_edge_bandwidth_usage_mut_proba=0.1, 
            path_transponder_assignment_mut_proba=0.9
        )
        assert gm.path_edge_bandwidth_usage_mut_proba == 0.1
        assert gm.path_transponder_assignment_mut_proba == 0.9

    def test_elitism_preservation(self, mutator, tagged_population):
        """
        Verify that the elite_size individuals are returned exactly as they were.
        """
        aggressive_mutator = GeneMutation(
            path_edge_bandwidth_usage_mut_proba=1.0, 
            path_transponder_assignment_mut_proba=1.0
        )
        elite_size = 1
        
        original_elite_val = tagged_population.path_edge_bandwidth_usage[0, 0, 1, 0, 1].item()
        new_pop = aggressive_mutator.mutate(tagged_population, elite_size)
        
        # Elite (Index 0) preserved
        assert new_pop.path_edge_bandwidth_usage[0, 0, 1, 0, 1] == original_elite_val
        
        # Non-Elite (Index 1) mutated
        new_non_elite_val = new_pop.path_edge_bandwidth_usage[1, 0, 1, 0, 1].item()
        assert new_non_elite_val != original_elite_val + 1.0  # Original was 101.0

    @patch("net_opt.core.mutations.gene_mutation.Bernoulli")
    def test_mutation_logic_deterministic(self, MockBernoulli, mutator, tagged_population):
        """
        Mock Bernoulli and Poisson to force specific outcomes and verify logic.
        """
        elite_size = 2
        
        # Force mutation on all elements
        mock_dist = MagicMock()
        MockBernoulli.return_value = mock_dist
        mock_dist.sample.side_effect = lambda size: torch.ones(size)

        # Force Poisson to return 999.0
        def poisson_side_effect(input_tensor):
            return torch.ones_like(input_tensor) * 999.0
            
        with patch("net_opt.core.mutations.gene_mutation.torch.poisson", side_effect=poisson_side_effect):
            # Patch limits to 10000.0 to prevent clamping interfering with the check
            with patch("net_opt.core.population.Population.edge_size_limits", new_callable=PropertyMock) as mock_limits:
                mock_limits.return_value = torch.tensor(10000.0) 
                
                new_pop = mutator.mutate(tagged_population, elite_size)
        
        # Elite (Index 0) unchanged (100.0)
        assert new_pop.path_edge_bandwidth_usage[0, 0, 1, 0, 1] == 100.0

        # Non-Elite (Index 2) mutated to forced value (999.0)
        assert new_pop.path_edge_bandwidth_usage[2, 0, 1, 0, 1] == 999.0

    def test_clamping_logic(self, mutator, tagged_population):
        """
        Verify that mutated values do not exceed edge_size_limits.
        """
        mutator.path_edge_bandwidth_usage_mut_proba = 1.0
        elite_size = 0
        
        # Inject huge values (5000.0)
        def huge_poisson(tensor):
            return torch.ones_like(tensor) * 5000.0

        with patch("net_opt.core.mutations.gene_mutation.torch.poisson", side_effect=huge_poisson):
            new_pop = mutator.mutate(tagged_population, elite_size)
            
            val = new_pop.path_edge_bandwidth_usage[0, 0, 1, 0, 1]
            # Should be clamped to limit (~96.0)
            assert val < 200.0