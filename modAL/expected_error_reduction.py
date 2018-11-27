"""
Expected error reduction framework for active learning.
"""

from typing import Tuple

import numpy as np

from scipy.stats import entropy
from sklearn.base import clone

from modAL.models import ActiveLearner
from modAL.utils.data import modALinput, data_vstack
from modAL.utils.selection import multi_argmax


def expected_error_reduction(classifier: ActiveLearner, X: modALinput,
                             p_subsample=1.0: np.float, n_instances=1: int) -> Tuple[np.ndarray, modALinput]:

    expected_error = np.full(shape=(len(X), ), fill_value=-np.nan)
    possible_labels = np.unique(classifier.y_training)

    for x_idx, x in enumerate(X):
        # subsample the data if needed
        if np.random.rand() <= p_subsample:
            # estimate the expected error
            for y in possible_labels:
                X_new = data_vstack((classifier.X_training, x))
                y_new = None

                refitted_estimator = clone(classifier.estimator).fit()


    query_idx = multi_argmax(expected_error, n_instances)

    return query_idx, X[query_idx]

