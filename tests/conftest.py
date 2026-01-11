import pytest
import torch

from net_opt.core.population import Population

@pytest.fixture
def sample_data():
    """
    Generates sample data parameters for testing.
    """
    P, N, T = 2, 3, 4
    
    encrypted_mat = torch.randint(0, 2, (N, N)).bool()
    
    bw_usage = torch.ones(P, N, N, N, N)
    transponder = torch.ones(P, N, N, T)
    
    return {
        "encrypted": encrypted_mat,
        "bw": bw_usage,
        "transponder": transponder,
        "P": P, "N": N, "T": T
    }

@pytest.fixture
def population_factory(sample_data):
    """
    A factory fixture that returns a function to create Population instances.
    Now accepts 'path_transponder_assignment' as an optional argument.
    """
    def _create(bandwidth_usage=None, path_transponder_assignment=None, P=None, N=None, T=None):
        # Use defaults from sample_data if not provided
        _P = P or sample_data["P"]
        _N = N or sample_data["N"]
        _T = T or sample_data["T"]
        
        if bandwidth_usage is not None:
            _P = bandwidth_usage.shape[0]
            _N = bandwidth_usage.shape[1]
        else:
            bandwidth_usage = sample_data["bw"]

        if path_transponder_assignment is None:
            # Create zeros with the correct shape
            path_transponder_assignment = torch.zeros(_P, _N, _N, _T)

        encrypted_mat = torch.zeros((_N, _N), dtype=torch.bool)

        return Population.masked(
            encrypted_neigh_matrix=encrypted_mat,
            path_edge_bandwidth_usage=bandwidth_usage,
            path_transponder_assignment=path_transponder_assignment
        )
    return _create

@pytest.fixture
def mock_population(sample_data, population_factory):
    """Standard mock population using default sample data."""
    return population_factory()