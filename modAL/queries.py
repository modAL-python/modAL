import numpy as np


def max_utility(utility, n_instances=1):
    query_idx = np.argpartition(-utility, n_instances)[:n_instances]
    return query_idx