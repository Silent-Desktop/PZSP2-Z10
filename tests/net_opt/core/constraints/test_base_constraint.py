import pytest
import torch
from jaxtyping import Float
from torch import Tensor
from net_opt.core.constraints.base_constraint import Constraint
from net_opt.core.population import Population


class ConcreteConstraint(Constraint):
    """
    A minimal concrete implementation of Constraint to allow testing 
    of the non-abstract methods.
    """
    def check(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        return torch.zeros(population.path_edge_bandwidth_usage.shape[0])

    def _check_all(self, population: Population, transponder_capacities: Float[Tensor, "T"], demand: Float[Tensor, "N N"]) -> Float[Tensor, "P"]:
        return torch.zeros(population.path_edge_bandwidth_usage.shape[0])


class TestConstraint:
    @pytest.fixture
    def constraint(self):
        """Returns an instance of the concrete constraint."""
        return ConcreteConstraint(readable_name="Test Constraint", weight=1.5)

    def test_constraint_instantiation(self, constraint):
        """Test valid instantiation and Pydantic field validation."""
        assert constraint.readable_name == "Test Constraint"
        assert constraint.weight == 1.5

    def test_abstract_class_error(self):
        """Ensure the base ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Constraint(readable_name="Abstract", weight=1.0) # type: ignore

    def test_mean_similarity_scores_logic(self, constraint):
        """
        Verifies that scores are averaged correctly across all non-batch dimensions.
        """
        # Shape (P=2, D1=2, D2=2)
        # Pop 0: All 0.5 -> Mean 0.5
        # Pop 1: All 1.0 -> Mean 1.0
        scores = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0, 1.0], [1.0, 1.0]]
        ])
        
        result = constraint._mean_similarity_scores(scores)
        
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(0.5))
        assert torch.isclose(result[1], torch.tensor(1.0))


    def test_calculate_similarity_scores_all_basic(self, constraint):
        """
        Test element-wise calculation: diff / max(a, b).
        """
        # P=1, D=2
        a = torch.tensor([[10.0, 20.0]])
        b = torch.tensor([[5.0,  20.0]])
        
        # Idx 0: |10-5| / 10 = 0.5
        # Idx 1: |20-20| / 20 = 0.0
        
        result = constraint._calculate_similarity_scores_all(a, b)
        
        expected = torch.tensor([[0.5, 0.0]])
        assert torch.allclose(result, expected)

    def test_calculate_similarity_scores_all_zero_division(self, constraint):
        """Test that 0/0 results in 0.0 via nan_to_num."""
        a = torch.tensor([0.0])
        b = torch.tensor([0.0])
        
        result = constraint._calculate_similarity_scores_all(a, b)
        
        assert result.item() == 0.0
        assert not torch.isnan(result).any()

    def test_calculate_similarity_scores_all_custom_transform(self, constraint):
        """Test passing a custom diff_transform (e.g., ReLU)."""
        a = torch.tensor([10.0])
        b = torch.tensor([15.0])
        
        # Using ReLU: relu(10 - 15) = 0.
        # Result: 0 / 15 = 0.0
        result = constraint._calculate_similarity_scores_all(
            a, b, diff_transform=torch.nn.functional.relu
        )
        assert result.item() == 0.0


    def test_calculate_similarity_scores_from_sums_logic(self, constraint):
        """
        Test aggregation logic: sum(diff) / sum(max(a,b)) per individual.
        """
        # P=2, D=2
        # Pop 0: a=[10, 10], b=[5, 5]. 
        #   Diff sum = 5+5=10. Goal sum = 10+10=20. Ratio = 0.5.
        # Pop 1: a=[100, 0], b=[100, 0].
        #   Diff sum = 0+0=0. Goal sum = 100+0=100. Ratio = 0.0.
        
        a = torch.tensor([[10.0, 10.0], [100.0, 0.0]])
        b = torch.tensor([[5.0,  5.0],  [100.0, 0.0]])
        
        result = constraint._calculate_similarity_scores_from_sums(a, b)
        
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(0.5))
        assert torch.isclose(result[1], torch.tensor(0.0))

    def test_calculate_similarity_scores_from_sums_zero_division(self, constraint):
        """Test that if the sum of goals is 0, we get 0 instead of NaN."""
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[0.0, 0.0]])
        
        result = constraint._calculate_similarity_scores_from_sums(a, b)
        
        assert result.item() == 0.0
        assert not torch.isnan(result).any()


    def test_methods_accept_complex_shapes(self, constraint):
        """
        Verify methods handle high-dimensional tensors (like Population 5D tensors) 
        without crashing.
        """
        # 5D input: (P=2, 2, 2, 2, 2)
        a = torch.rand(2, 2, 2, 2, 2)
        b = torch.rand(2, 2, 2, 2, 2)
        
        # Test _calculate_similarity_scores_all (should preserve shape)
        res_all = constraint._calculate_similarity_scores_all(a, b)
        assert res_all.shape == a.shape
        
        # Test _calculate_similarity_scores_from_sums (should reduce to P)
        res_sums = constraint._calculate_similarity_scores_from_sums(a, b)
        assert res_sums.shape == (2,)