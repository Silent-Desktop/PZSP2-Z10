import pytest
import torch

@pytest.fixture
def sample_data():
    """
    Generates sample data for testing
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