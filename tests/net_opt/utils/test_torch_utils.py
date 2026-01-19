import torch
import pytest

from net_opt.utils.torch_utils import multi_triu


class TestMultiTriu:
    """
    Test suite for the multi_triu function.
    """

    def test_basic_2d_masking(self):
        """
        Test standard 2D case.
        Logic: indices_a >= indices_b are masked (Lower Triangle + Diagonal).
        """
        x = torch.ones(3, 3)
        res = multi_triu(x, [(0, 1)], value=0.0)

        expected = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )

        torch.testing.assert_close(res, expected)

    def test_custom_value(self):
        """
        Ensure the masked area is filled with the specified value.
        """
        x = torch.ones(2, 2)
        fill_val = -5.0
        res = multi_triu(x, [(0, 1)], value=fill_val)

        expected = torch.tensor([[fill_val, 1.0], [fill_val, fill_val]])

        torch.testing.assert_close(res, expected)

    def test_no_dim_pairs(self):
        """
        If dim_pairs is empty, return a clone of the original tensor.
        """
        x = torch.randn(3, 4, 5)
        res = multi_triu(x, [])

        assert res is not x  # Must be a new object (clone)
        torch.testing.assert_close(res, x)

    def test_broadcasting_batch_dim(self):
        """
        Test 3D tensor where the first dim is batch.
        Shape (B, H, W). Pair (1, 2) masks H vs W.
        """
        B, H, W = 2, 3, 3
        x = torch.ones(B, H, W)
        res = multi_triu(x, [(1, 2)], value=0.0)

        expected_slice = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )

        for b in range(B):
            torch.testing.assert_close(res[b], expected_slice)

    def test_multiple_dim_pairs_intersection(self):
        """
        Test logic where multiple pairs are provided.
        """
        x = torch.ones(2, 2, 2)

        pairs = [(0, 1), (1, 2)]
        res = multi_triu(x, pairs, value=0.0)

        assert res[1, 1, 0] == 0.0
        assert res[0, 0, 1] == 1.0

    def test_index_out_of_bounds(self):
        """
        Should raise IndexError if dims are invalid.
        """
        x = torch.zeros(3, 3)
        with pytest.raises(IndexError, match="Dimension index out of range"):
            multi_triu(x, [(0, 5)])

    def test_non_square_inputs(self):
        """
        Test behavior on rectangular matrices.
        """
        x = torch.ones(2, 3)
        res = multi_triu(x, [(0, 1)], value=0)

        expected = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(res, expected)
