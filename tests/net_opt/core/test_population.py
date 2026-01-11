import torch

from net_opt.core.population import Population

class TestPopulation:
    def test_masked_method_bandwidth_usage(self, sample_data):
        """
        Test that .masked() correctly zeroes out the diagonal elements 
        of path_edge_bandwidth_usage.
        """
        enc = sample_data["encrypted"]
        bw = sample_data["bw"]
        trans = sample_data["transponder"]
        N = sample_data["N"]
        
        # Before masking
        assert bw[0, 0, 0, 0, 0] == 1.0
        
        pop = Population.masked(enc, bw, trans)
        res_bw = pop.path_edge_bandwidth_usage
        
        for i in range(N):
            # Diagonal
            assert torch.all(res_bw[:, :, :, i, i] == 0.0)
            
            # Off diagonal
            if i > 0:   
                assert torch.all(res_bw[:, :, :, i, i-1] == 1.0)

    def test_masked_method_transponders(self, sample_data):
        """
        Test that .masked() applies multi_triu to path_transponder_assignment
        on dimensions 1 and 2 (the N x N grid).
        """
        enc = sample_data["encrypted"]
        bw = sample_data["bw"]
        trans = sample_data["transponder"]
        
        pop = Population.masked(enc, bw, trans)
        res_trans = pop.path_transponder_assignment
        
        # Low triangle
        assert res_trans[0, 1, 0, 0] == 0.0 
        assert res_trans[0, 1, 1, 0] == 0.0 

        # Up triangle
        assert res_trans[0, 0, 1, 0] == 1.0 

    def test_edge_size_limits_property(self):
        """
        Test that edge_size_limits correctly calculates bandwidth 
        based on the boolean encrypted matrix.
        """
        N = 2
        enc_matrix = torch.tensor([[True, False], [False, True]])
        
        P, T = 1, 1
        bw = torch.zeros(P, N, N, N, N)
        trans = torch.zeros(P, N, N, T)
        
        pop = Population(
            encrypted_neigh_matrix=enc_matrix,
            path_edge_bandwidth_usage=bw,
            path_transponder_assignment=trans,
            regular_bandwidth=100,
            encrypted_bandwidth=50
        )
        
        limits = pop.edge_size_limits

        expected = torch.tensor([
            [50., 100.],
            [100., 50.]
        ])
        
        torch.testing.assert_close(limits, expected)