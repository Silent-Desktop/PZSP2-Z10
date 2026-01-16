import pytest
import torch

from net_opt.core.population import Population


@pytest.fixture
def sample_data():
    """
    Generates sample data parameters for testing.
    """
    encrypted_width, regular_width = 30, 96
    P, N, T = 2, 3, 4

    encrypted_mat = torch.randint(0, 2, (N, N)).bool()

    bw_usage = torch.ones(P, N, N, N, N)
    transponder = torch.ones(P, T, N, N)
    return {
        "encrypted": encrypted_mat,
        "bw": bw_usage,
        "transponder": transponder,
        "P": P,
        "N": N,
        "T": T,
        "encrypted_bandwidth": encrypted_width,
        "regular_bandwidth": regular_width,
    }


@pytest.fixture
def population_factory(sample_data):
    """
    A factory fixture that returns a function to create Population instances.
    Now accepts 'path_transponder_assignment' as an optional argument.
    """

    def _create(
        bandwidth_usage=None,
        path_transponder_assignment=None,
        P=None,
        N=None,
        T=None,
        neigh_matrix=None,
        encrypted_bandwidth=None,
        regular_bandwidth=None
    ):
        # Use defaults from sample_data if not provided
        _P = P or sample_data["P"]
        _N = N or sample_data["N"]
        _T = T or sample_data["T"]
        _e = encrypted_bandwidth or sample_data["encrypted_bandwidth"]
        _r = regular_bandwidth or sample_data["regular_bandwidth"]

        if bandwidth_usage is not None:
            _P = bandwidth_usage.shape[0]
            _N = bandwidth_usage.shape[1]
        else:
            bandwidth_usage = sample_data["bw"]

        if path_transponder_assignment is None:
            # Create zeros with the correct shape
            path_transponder_assignment = torch.zeros(_P, _T, _N, _N)

        if neigh_matrix is None:
            neigh_matrix = ~torch.eye(_N, dtype=torch.bool)

        encrypted_mat = torch.zeros((_N, _N), dtype=torch.bool)

        return Population.masked(
            encrypted_neigh_matrix=encrypted_mat,
            path_edge_bandwidth_usage=bandwidth_usage,
            path_transponder_assignment=path_transponder_assignment,
            encrypted_bandwidth=_e,
            regular_bandwidth=_r,
            neigh_matrix=neigh_matrix,
        )

    return _create


@pytest.fixture
def mock_population(sample_data, population_factory):
    """Standard mock population using default sample data."""
    return population_factory()
