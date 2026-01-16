import torch

from net_opt.core.population import Population


class TestPopulation:
    def test_masked_method_bandwidth_usage(self, sample_data):
        """
        Test that .masked() correctly zeroes out the diagonal and tril elements
        of path_edge_bandwidth_usage.
        """
        enc = sample_data["encrypted"]
        bw = sample_data["bw"]
        trans = sample_data["transponder"]
        N = sample_data["N"]
        e = sample_data["encrypted_bandwidth"]
        r = sample_data["regular_bandwidth"]

        # Before masking
        assert torch.all(bw == 1.0)

        pop = Population.masked(
            enc, bw, trans, e, r, ~torch.eye(N, dtype=torch.bool)
        )
        res_bw = pop.path_edge_bandwidth_usage

        # diagonal
        d = torch.arange(N)
        assert torch.all(res_bw[:, d, d, :, :] == 0.0)
        # tril
        i, j = torch.tril_indices(N, N)
        assert torch.all(res_bw[:, :, :, i, j] == 0.0)
        # go back to all ones
        res_bw[:, d, d, :, :] = 1.0
        res_bw[:, :, :, i, j] = 1.0
        # check if nothing extra got masked
        assert torch.all(res_bw == 1.0)

    def test_masked_method_transponders(self, sample_data):
        """
        Test that .masked() applies triu to path_transponder_assignment
        on dimensions 2 and 3 (the N x N grid).
        """
        enc = sample_data["encrypted"]
        bw = sample_data["bw"]
        trans = sample_data["transponder"]
        e = sample_data["encrypted_bandwidth"]
        r = sample_data["regular_bandwidth"]

        pop = Population.masked(
            enc, bw, trans, e, r, ~torch.eye(enc.size(0), dtype=torch.bool)
        )
        res_trans = pop.path_transponder_assignment

        # Low triangle
        assert res_trans[0, 0, 1, 0] == 0.0
        assert res_trans[0, 0, 1, 1] == 0.0

        # Up triangle
        assert res_trans[0, 0, 0, 1] == 1.0

    def test_edge_size_limits_property(self):
        """
        Test that edge_size_limits correctly calculates bandwidth
        based on the boolean encrypted matrix.
        """
        N = 2
        enc_matrix = torch.tensor([[True, False], [False, True]])

        P, T = 1, 1
        bw = torch.zeros(P, N, N, N, N)
        trans = torch.zeros(P, T, N, N)

        pop = Population(
            encrypted_neigh_matrix=enc_matrix,
            path_edge_bandwidth_usage=bw,
            path_transponder_assignment=trans,
            regular_bandwidth=100,
            encrypted_bandwidth=50,
            neigh_matrix=~torch.eye(N, dtype=torch.bool),
        )

        limits = pop.edge_size_limits

        expected = torch.tensor([[50.0, 100.0], [100.0, 50.0]])

        torch.testing.assert_close(limits, expected)
