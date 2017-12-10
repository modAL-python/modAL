import numpy as np


def max_utility(utility, n_instances=1):
    """
    Selects n_instances instance having the highest utility

    Parameters
    ----------
    utility: array-like, shape = (n_instances, 1)
        Contains the utility values for the instances

    n_instances: int
        Specifies how many largest values to return

    Returns
    -------
    query_idx: numpy.ndarray, shape = (n_instances, 1)
        Contains the indices of the n_instances largest
        values in utility

    """
    assert n_instances <= len(utility), 'n_instances must be less or equal than the size of utility'

    query_idx = np.argpartition(-utility, n_instances)[:n_instances]
    return query_idx


def utility_weighted_random(utility, n_instances=1):
    assert n_instances <= len(utility), 'n_instances must be less or equal than the size of utility'

    query_idx = np.random.choice(range(len(utility)), size=n_instances, p=utility/np.sum(utility), replace=False)
    return query_idx
