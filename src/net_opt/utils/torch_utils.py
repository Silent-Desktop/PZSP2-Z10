import torch
from torch import Tensor
import functools

def multi_triu(
    tensor: torch.Tensor,
    dim_pairs: list[tuple[int, int]],
    value: float = 0.0
) -> torch.Tensor:
    """
    Multiple triu between dim pairs
    """
    if not dim_pairs:
        return tensor.clone()
        
    shape = tensor.shape
    ndim = tensor.ndim

    all_masks: list[Tensor] = []
    for dim_a, dim_b in dim_pairs:
        if not (0 <= dim_a < ndim and 0 <= dim_b < ndim):
            raise IndexError(f"Dimension index out of range. Tensor has {ndim} dims, got pair ({dim_a}, {dim_b})")

        size_a = shape[dim_a]
        size_b = shape[dim_b]
        
        view_shape_a = [1] * ndim
        view_shape_a[dim_a] = size_a
        indices_a = torch.arange(size_a).view(view_shape_a)
        
        view_shape_b = [1] * ndim
        view_shape_b[dim_b] = size_b
        indices_b = torch.arange(size_b).view(view_shape_b)

        pair_mask = (indices_a >= indices_b)
        all_masks.append(pair_mask)
        
    final_mask = functools.reduce(torch.logical_and, all_masks)
    
    tensor[final_mask.expand_as(tensor)] = value
    return tensor