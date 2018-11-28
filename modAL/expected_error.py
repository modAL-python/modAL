"""
Expected error reduction framework for active learning.
"""

from typing import Tuple

import numpy as np

from scipy.stats import entropy

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from modAL.models import ActiveLearner
from modAL.utils.data import modALinput, data_vstack
from modAL.utils.selection import multi_argmax


def expected_error_reduction(learner: ActiveLearner, X: modALinput,
                             p_subsample: np.float = 1.0, n_instances: int = 1) -> Tuple[np.ndarray, modALinput]:
    """
    Expected error reduction query strategy.

    References:
        Roy and McCallum, 2001 (http://groups.csail.mit.edu/rrg/papers/icml01.pdf)

    Args:
        learner: The ActiveLearner object for which the expected error is to be estimated.
        X: The samples.
        p_subsample: Probability of keeping a sample from the pool when calculating expected error.
            Significantly improves runtime for large sample pools.
        n_instances: The number of instances to be sampled.


    Returns:
        The indices of the instances from X chosen to be labelled; the instances from X chosen to be labelled.
    """

    assert 0.0 <= p_subsample <= 1.0, 'p_subsample subsampling keep ratio must be between 0.0 and 1.0'

    #expected_error = np.full(shape=(len(X), ), fill_value=-np.nan)
    expected_error = np.zeros(shape=(len(X), ))
    possible_labels = np.unique(learner.y_training)

    try:
        X_proba = learner.predict_proba(X)
    except NotFittedError:
        # TODO: implement a proper cold-start
        return 0, X[0]

    for x_idx, x in enumerate(X):
        # subsample the data if needed
        if np.random.rand() <= p_subsample:
            # estimate the expected error
            for y_idx, y in enumerate(possible_labels):
                X_new = data_vstack((learner.X_training, x.reshape(1, -1)))
                y_new = data_vstack((learner.y_training, np.array(y).reshape(1, )))

                refitted_estimator = clone(learner.estimator).fit(X_new, y_new)
                uncertainty = 1 - np.max(refitted_estimator.predict_proba(X), axis=1)

                expected_error[x_idx] += np.sum(uncertainty)*X_proba[x_idx, y_idx]

        else:
            expected_error[x_idx] -np.nan

    query_idx = multi_argmax(expected_error, n_instances)

    return query_idx, X[query_idx]

