import typing
from functools import singledispatch

import torch
import numpy as np


def is_binary(input: torch.Tensor) -> bool:
    """
    Check whether the input contains only 0s and/or 1s
    Args:
        input: Tensor to check
    Returns:
        bool: True if contains only 0s and/or 1s, False otherwise

    """
    zero_and_one = torch.tensor([0., 1.], device=input.device)
    return all([value in zero_and_one for value in torch.unique(input)])


def has_only_zeros(input: np.ndarray) -> bool:
    """
    Check whether the input contains only 0s.
    Args:
        input: Array to check

    Returns:
        bool: True if contains only 0s, False otherwise

    """
    return not np.any(input != 0)


def calculate_intersection(x1: torch.Tensor, x2: torch.Tensor,
                           dim=None) -> torch.Tensor:
    """
    Calculates intersection of two tensors.
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimensions along which the intersection will be summed
    Returns:
        torch.Tensor: Tensor containing the intersection. Dimensionality
            may differ, depending on the dim argument
    """
    return (x1 * x2).sum(dim=dim) if dim is not None else (x1 * x2).sum()


def calculate_union(x1: torch.Tensor, x2: torch.Tensor,
                    dim=None) -> torch.Tensor:
    """
    Calculates the union of two tensors
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimensions along which the union will be summed
    Returns:
        torch.Tensor: Tensor containing the union. Dimensionality may
            differ, depending on the dim argument
    """
    return x1.sum(dim=dim) + x2.sum(dim=dim) if dim is not None \
        else x1.sum() + x2.sum()


def calculate_false_negatives(prediction: torch.Tensor, target: torch.Tensor,
                              dim=None) -> torch.Tensor:
    """
    Calculate false negatives
    Args:
        prediction: Network output.
        target: Target values.
        dim: Dimensions along which the FNs will be summed.
    Returns:
        torch.Tensor: Number of false negatives
    """
    return (target * (1 - prediction)).sum(dim=dim) if dim is not None else (
            target * (1 - prediction)).sum()


def calculate_false_positives(prediction: torch.Tensor, target: torch.Tensor,
                              dim=None) -> torch.Tensor:
    """
    Calculate false positives
    Args:
        prediction: Network output.
        target: Target values.
        dim: Dimensions along which the FPs will be summed.
    Returns:
        torch.Tensor: Number of false positives
    """
    return ((1 - target) * prediction).sum(dim=dim) if dim is not None else (
            (1 - target) * prediction).sum()


@singledispatch
def copy(tensor: typing.Union[np.ndarray, torch.Tensor]):
    """
        Function that returns copy of given input.

    """
    raise TypeError(f"Type {type(tensor)} not supported")


@copy.register(np.ndarray)
def _(tensor: np.ndarray) -> np.ndarray:
    return np.copy(tensor)


@copy.register(torch.Tensor)
def _(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()
