"""
Functions to select certain element indices from arrays.
"""

import numpy as np


def multi_argmax(values, n_instances=1):
    """
    Selects the indices of the n_instances highest values.

    Parameters
    ----------
    values: numpy.ndarray of shape = (n_samples, 1)
        Contains the values to be selected from.

    n_instances: int
        Specifies how many indices to return.

    Returns
    -------
    max_idx: numpy.ndarray of shape = (n_samples, 1)
        Contains the indices of the n_instances largest values.

    """
    assert n_instances <= len(values), 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances-1)[:n_instances]
    return max_idx


def weighted_random(weights, n_instances=1):
    """
    Returns n_instances indices based on the weights.

    Parameters
    ----------
    weights: numpy.ndarray of shape = (n_samples, 1)
        Contains the weights of the sampling.

    n_instances: int
        Specifies how many indices to return.

    Returns
    -------
    random_idx: numpy.ndarray of shape = (n_instances, 1)
        n_instances random indices based on the weights.
    """
    assert n_instances <= len(weights), 'n_instances must be less or equal than the size of utility'

    random_idx = np.random.choice(range(len(weights)), size=n_instances, p=weights / np.sum(weights), replace=False)
    return random_idx
