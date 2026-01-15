from torch.distributions import Categorical, Poisson
import torch


def get_opt_init_dist(
    N: int, T: int, encrypted_bandwidth: int, regular_bandwidth: int
) -> tuple[Categorical, Categorical]:
    """Returns initial value distributions with total expected values that shouldn't breach limit constraints."""
    mu = regular_bandwidth // 2
    N_pairs = N * (N - 1)
    skip = mu // N_pairs
    p = mu / N_pairs - skip
    probs = [0] * (skip) + [1 - p] + [p]
    path_dist = Categorical(torch.tensor(probs))
    trans_dist = Poisson(encrypted_bandwidth // T)
    return path_dist, trans_dist
