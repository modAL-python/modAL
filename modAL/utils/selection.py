"""
Functions to select certain element indices from arrays.
"""

import numpy as np


def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances highest values.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.

    Returns:
        Contains the indices of the n_instances largest values.
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
