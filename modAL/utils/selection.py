"""
Functions to select certain element indices from arrays.
"""

import numpy as np


def shuffled_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Shuffles the values and sorts them afterwards. This can be used to break
    the tie when the highest utility score is not unique. The shuffle randomizes
    order, which is preserved by the mergesort algorithm.

    Args:
        values:
        n_instances:

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.

    Returns:
        The indices of the n_instances largest values.
    """

    # shuffling indices and corresponding values
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]

    # getting the n_instances best instance
    # since mergesort is used, the shuffled order is preserved
    sorted_query_idx = np.argsort(shuffled_values, kind='mergesort')[:n_instances]

    # inverting the shuffle
    query_idx = shuffled_idx[sorted_query_idx]
    return query_idx


def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances highest values.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.

    Returns:
        The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx


def weighted_random(weights: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Returns n_instances indices based on the weights.

    Args:
        weights: Contains the weights of the sampling.
        n_instances: Specifies how many indices to return.

    Returns:
        n_instances random indices based on the weights.
    """
    assert n_instances <= weights.shape[0], 'n_instances must be less or equal than the size of utility'
    weight_sum = np.sum(weights)
    assert weight_sum > 0, 'the sum of weights must be larger than zero'

    random_idx = np.random.choice(range(len(weights)), size=n_instances, p=weights/weight_sum, replace=False)
    return random_idx
