"""
Expected error reduction framework for active learning.
"""

from typing import Tuple

import numpy as np

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from modAL.models import ActiveLearner
from modAL.utils.data import modALinput, data_vstack
from modAL.utils.selection import multi_argmax, shuffled_argmax
from modAL.uncertainty import _proba_uncertainty, _proba_entropy


def expected_error_reduction(learner: ActiveLearner, X: modALinput, loss: str = 'binary',
                             p_subsample: np.float = 1.0, n_instances: int = 1,
                             random_tie_break: bool = False) -> Tuple[np.ndarray, modALinput]:
    """
    Expected error reduction query strategy.

    References:
        Roy and McCallum, 2001 (http://groups.csail.mit.edu/rrg/papers/icml01.pdf)

    Args:
        learner: The ActiveLearner object for which the expected error
            is to be estimated.
        X: The samples.
        loss: The loss function to be used. Can be 'binary' or 'log'.
        p_subsample: Probability of keeping a sample from the pool when
            calculating expected error. Significantly improves runtime
            for large sample pools.
        n_instances: The number of instances to be sampled.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.


    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """

    assert 0.0 <= p_subsample <= 1.0, 'p_subsample subsampling keep ratio must be between 0.0 and 1.0'
    assert loss in ['binary', 'log'], 'loss must be \'binary\' or \'log\''

    expected_error = np.zeros(shape=(len(X), ))
    possible_labels = np.unique(learner.y_training)

    try:
        X_proba = learner.predict_proba(X)
    except NotFittedError:
        # TODO: implement a proper cold-start
        return 0, X[0]

    cloned_estimator = clone(learner.estimator)

    for x_idx, x in enumerate(X):
        # subsample the data if needed
        if np.random.rand() <= p_subsample:
            # estimate the expected error
            for y_idx, y in enumerate(possible_labels):
                X_new = data_vstack((learner.X_training, x.reshape(1, -1)))
                y_new = data_vstack((learner.y_training, np.array(y).reshape(1, )))

                cloned_estimator.fit(X_new, y_new)
                refitted_proba = cloned_estimator.predict_proba(X)
                if loss is 'binary':
                    loss = _proba_uncertainty(refitted_proba)
                elif loss is 'log':
                    loss = _proba_entropy(refitted_proba)

                expected_error[x_idx] += np.sum(loss)*X_proba[x_idx, y_idx]

        else:
            expected_error[x_idx] = np.inf

    if not random_tie_break:
        query_idx = multi_argmax(expected_error, n_instances)
    else:
        query_idx = shuffled_argmax(expected_error, n_instances)

    return query_idx, X[query_idx]
