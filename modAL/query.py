"""
Query strategies for the active learning model.
"""

import numpy as np


def max_uncertainty(uncertainty, n_samples=1):
    """
    Selects n_samples samples having the highest utility

    Parameters
    ----------
    uncertainty: array-like, shape = (n_samples, 1)
        Contains the utility values for the instances

    n_samples: int
        Specifies how many largest values to return

    Returns
    -------
    query_idx: numpy.ndarray, shape = (n_samples, 1)
        Contains the indices of the n_samples largest
        values in uncertainties

    """
    assert n_samples <= len(uncertainty), 'n_instances must be less or equal than the size of utility'

    query_idx = np.argpartition(-uncertainty, n_samples)[:n_samples]
    return query_idx


def uncertainty_weighted_random(uncertainty, n_samples=1):
    """
    Samples n_samples samples, using uncertainties as weights

    Parameters
    ----------
    uncertainty: array-like, shape = (n_samples, 1)
        Contains the uncertainty values for the instances

    n_samples: int
        Specifies how many largest values to return

    Returns
    -------
    query_idx: numpy.ndarray, shape = (n_samples, 1)
        Contains the indices of the n_samples sample
        given by the query
    """
    assert n_samples <= len(uncertainty), 'n_instances must be less or equal than the size of utility'

    query_idx = np.random.choice(range(len(uncertainty)), size=n_samples, p=uncertainty / np.sum(uncertainty), replace=False)
    return query_idx
