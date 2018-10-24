import numpy as np

from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from typing import Tuple


def SVM_binary_minimum(classifier: BaseEstimator, X_pool: modALinput) -> Tuple[np.ndarray, modALinput]:
    """
    SVM binary minimum multilabel active learning strategy. For details see the paper
    Klaus Brinker, On Active Learning in Multi-label Classification
    (https://link.springer.com/chapter/10.1007%2F3-540-31314-1_24)

    Args:
        classifier: The multilabel classifier for which the labels are to be queried. Must be an SVM model
        such as the ones from sklearn.svm.
        X: The pool of samples to query from.

    Returns:
        The index of the instance from X chosen to be labelled; the instance from X chosen to be labelled.
    """
    min_abs_dist = np.min(np.abs(classifier.estimator.decision_function(X_pool)), axis=1)
    query_idx = np.argmin(min_abs_dist)
    return query_idx, X_pool[query_idx]