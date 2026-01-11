import pytest
import torch
from net_opt.core.constraints.path_edge_bandwidth_kirchhoff import PathEdgeBandwidthKirchhoff

class TestPathEdgeBandwidthKirchhoff:

    @pytest.fixture
    def constraint(self):
        return PathEdgeBandwidthKirchhoff()

    @pytest.fixture
    def empty_inputs(self, sample_data):
        return {
            "transponder_capacities": torch.zeros(sample_data["T"]),
            "demand": torch.zeros(sample_data["N"], sample_data["N"])
        }

    def test_direct_link_ignored(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Path 0->1.
        Intermediate nodes: Node 2 (with 0 flow).
        Expected: 0.0 score (Perfect).
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Path 0->1: Flow on link 0->1
        usage[0, 0, 1, 0, 1] = 100.0
    
        
        pop = population_factory(bandwidth_usage=usage)
        results = constraint._check_all(pop, **empty_inputs)
        
        # Check Node 2 on Path 0->1
        assert results[0, 0, 1, 2] == 0.0
        # Check masked nodes are 0.0
        assert results[0, 0, 1, 0] == 0.0
        assert results[0, 0, 1, 1] == 0.0

    def test_perfect_multihop_flow(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Path 0->2 via Node 1.
        Flow 0->1 matches Flow 1->2.
        Expected: Node 1 (intermediate) has score 0.0.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Path 0->2
        # Link 0->1: 100 units
        usage[0, 0, 2, 0, 1] = 100.0
        # Link 1->2: 100 units
        usage[0, 0, 2, 1, 2] = 100.0

        pop = population_factory(bandwidth_usage=usage)
        results = constraint._check_all(pop, **empty_inputs)
        
        assert results[0, 0, 2, 1] == 0.0

    def test_broken_multihop_flow_leak(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Path 0->2 via Node 1.
        Flow enters Node 1 (100) but less leaves (40).
        Expected: Mismatch penalty at Node 1.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Path 0->2
        usage[0, 0, 2, 0, 1] = 100.0 # In to 1
        usage[0, 0, 2, 1, 2] = 40.0  # Out from 1
        
        # Diff = |100 - 40| = 60
        # Max = 100
        # Score = 0.6
        
        pop = population_factory(bandwidth_usage=usage)
        results = constraint._check_all(pop, **empty_inputs)
        
        assert torch.isclose(results[0, 0, 2, 1], torch.tensor(0.6))

    def test_broken_multihop_flow_gain(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Scenario: Flow appears out of nowhere at intermediate node.
        Flow 0->1 is 0, but Flow 1->2 is 50.
        Expected: Mismatch penalty at Node 1.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Path 0->2
        usage[0, 0, 2, 1, 2] = 50.0 
        # In to 1 is 0.
        
        # Diff = |0 - 50| = 50
        # Max = 50
        # Score = 1.0
        
        pop = population_factory(bandwidth_usage=usage)
        results = constraint._check_all(pop, **empty_inputs)
        
        assert results[0, 0, 2, 1] == 1.0

    def test_check_aggregation(self, constraint, population_factory, empty_inputs, sample_data):
        """
        Verify aggregation logic.
        Note: The return shape of _check_all is (P, N, N, N).
        The mean is calculated over N*N*N elements per population member.
        """
        P, N = sample_data["P"], sample_data["N"]
        usage = torch.zeros(P, N, N, N, N)
        
        # Pop 1: Single failure.
        # Path 0->2 via Node 1.
        # Node 1 has In=10, Out=0 -> Score 1.0.
        usage[1, 0, 2, 0, 1] = 10.0
        
        pop = population_factory(bandwidth_usage=usage)
        scores = constraint.check(pop, **empty_inputs)
        
        assert scores.shape == (P,)
        assert scores[0] == 0.0

        expected_score_p1 = 1.0 / (N * N * N) # Diluted penalty
        
        assert torch.isclose(scores[1], torch.tensor(expected_score_p1))