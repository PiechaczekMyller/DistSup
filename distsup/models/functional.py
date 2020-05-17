from typing import Callable

import torch
import numpy as np
import medpy.metric.binary as mp

import utils

BATCH_DIM = 0
FIRST_CLASS = 0


def dice(prediction: torch.Tensor,
         target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute an average DICE score across a batch of volumes for each class.
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values. The classes should be one-hot encoded.
            Dimensions - (Batch, Class, Depth, Height, Width)
        epsilon: Smooth factor for each element in the batch
    Returns:
        torch.Tensor: DICE score for each class averaged across the whole batch.
    """
    volume_dims = list(range(2, target.dim()))
    intersection = utils.calculate_intersection(prediction, target,
                                                dim=volume_dims)
    union = utils.calculate_union(prediction, target,
                                  dim=volume_dims)
    score = (2. * intersection) / (union + epsilon)
    score = torch.mean(score, dim=BATCH_DIM)
    return score


def recall(prediction: torch.Tensor,
           target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the recall score for each of the classes
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values. The classes should be one-hot encoded.
            Dimensions - (Batch, Class, Depth, Height, Width)
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: Recall score calculated for each class
            averaged across the whole batch
    """
    assert utils.is_binary(prediction), "Predictions must be binary"
    assert utils.is_binary(target), "Target must be binary"
    volume_dims = list(range(2, target.dim()))
    true_positives = utils.calculate_intersection(prediction, target,
                                                  dim=volume_dims)
    false_negatives = utils.calculate_false_negatives(prediction, target,
                                                      dim=volume_dims)
    score = true_positives / (
            true_positives + false_negatives + epsilon)
    score = torch.mean(score, dim=BATCH_DIM)
    return score


def precision(prediction: torch.Tensor,
              target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the precision score
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values. The classes should be one-hot encoded.
            Dimensions - (Batch, Class, Depth, Height, Width)
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: Precision score calculated for each class
            averaged across the whole batch
    """
    assert utils.is_binary(prediction), "Predictions must be binary"
    assert utils.is_binary(target), "Target must be binary"
    volume_dims = list(range(2, target.dim()))
    true_positives = utils.calculate_intersection(prediction, target,
                                                  dim=volume_dims)
    false_positive = utils.calculate_false_positives(prediction, target,
                                                     dim=volume_dims)
    score = true_positives / (
            true_positives + false_positive + epsilon)
    score = torch.mean(score, dim=BATCH_DIM)
    return score


def f_score(prediction: torch.Tensor,
            target: torch.Tensor, beta: float,
            epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the F score for each class
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values. The classes should be one-hot encoded.
            Dimensions - (Batch, Class, Depth, Height, Width)
        beta: Weight parameter between precision and recall
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: F Score calculated for each class
            averaged across the whole batch
    """
    precision_score = precision(prediction, target, epsilon)
    recall_score = recall(prediction, target, epsilon)
    score = (1 + beta ** 2) * ((precision_score * recall_score) / (
            beta ** 2 * precision_score + recall_score + epsilon))
    return score


def hausdorff95(prediction: torch.Tensor, target: torch.Tensor,
                merge_operation: Callable[
                    [torch.Tensor], float] = torch.max) -> torch.Tensor:
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD)
    between the binary objects in two images.
    Heavily depends on medpy.metric.binary.hd95, so check it out for more
    details.
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values.
            Dimensions - (Batch, Class, Depth, Height, Width)
        merge_operation: Operation to merge results from all the slices in
            the batch into a single value. Should be a function that accepts
            two dimensional tensors (Batch, Depth).
            Good examples are: torch.mean, torch.max, torch.min.
    Returns:
        torch.Tensor: Hausdorff distance for the provided volume
    """
    assert utils.is_binary(prediction), "Predictions must be binary"
    assert utils.is_binary(target), "Target must be binary"
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    volumes_count, _, slices_count, _, _ = prediction.shape
    results = np.zeros((volumes_count, slices_count))
    for volume_idx in range(volumes_count):
        for slice_idx in range(slices_count):
            prediction_slice = prediction[
                volume_idx, FIRST_CLASS, slice_idx, ...]
            target_slice = target[volume_idx, FIRST_CLASS, slice_idx, ...]
            if utils.has_only_zeros(prediction_slice) or \
                    utils.has_only_zeros(target_slice):
                results[volume_idx, slice_idx] = 0
            else:
                results[volume_idx, slice_idx] = mp.hd95(prediction_slice,
                                                         target_slice)
    return merge_operation(torch.from_numpy(results))
