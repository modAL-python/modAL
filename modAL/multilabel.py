import numpy as np

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier

from modAL.utils.data import modALinput
from typing import Tuple, Optional
from itertools import combinations


def _SVM_loss(multiclass_classifier: OneVsRestClassifier,
              X: modALinput,
              most_certain_classes: Optional[int] = None) -> np.ndarray:
    """
    Utility function for max_loss and mean_max_loss strategies.

    Args:
        multiclass_classifier: sklearn.multiclass.OneVsRestClassifier instance for which the loss
            is to be calculated.
        X: The pool of samples to query from.
        most_certain_classes: optional, indexes of most certainly predicted class for each instance.
            If None, loss is calculated for all classes.

    Returns:
        np.ndarray of shape (n_instances, ), losses for the instances in X.

    """
    predictions = 2*multiclass_classifier.predict(X)-1
    n_classes = len(multiclass_classifier.classes_)

    if most_certain_classes is None:
        cls_mtx = 2*np.eye(n_classes, n_classes) - 1
        loss_mtx = np.maximum(1-np.dot(predictions, cls_mtx), 0)
        return loss_mtx.mean(axis=1)
    else:
        cls_mtx = -np.ones(shape=(len(X), n_classes))
        for inst_idx, most_certain_class in enumerate(most_certain_classes):
            cls_mtx[inst_idx, most_certain_class] = 1

        cls_loss = np.maximum(1 - np.multiply(cls_mtx, predictions), 0).sum(axis=1)
        return cls_loss


def SVM_binary_minimum(classifier: BaseEstimator,
                       X_pool: modALinput) -> Tuple[np.ndarray, modALinput]:
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


def max_loss(classifier: BaseEstimator,
             X_pool: modALinput,
             n_instances: int = 1) -> Tuple[np.ndarray, modALinput]:

    """
    Max Loss query strategy for SVM multilabel classification.

    For more details on this query strategy, see
    Li et al., Multilabel SVM active learning for image classification
    (http://dx.doi.org/10.1109/ICIP.2004.1421535)

    Args:
        classifier: The multilabel classifier for which the labels are to be queried. Should be an SVM model
            such as the ones from sklearn.svm. Although the function will execute for other models as well,
            the mathematical calculations in Li et al. work only for SVM-s.
        X: The pool of samples to query from.

    Returns:
        The index of the instance from X chosen to be labelled; the instance from X chosen to be labelled.
    """

    most_certain_classes = classifier.predict_proba(X_pool).argmax(axis=1)
    loss = _SVM_loss(classifier, X_pool, most_certain_classes=most_certain_classes)

    assert len(X_pool) >= n_instances, 'n_instances cannot be larger than len(X_pool)'

    if n_instances == 1:
        query_idx = np.argmax(loss)
        return query_idx, X_pool[query_idx]
    else:
        max_val = -np.inf
        max_idx = None
        for subset_idx in combinations(range(len(X_pool)), n_instances):
            subset_sum = loss[list(subset_idx)].sum()
            if subset_sum > max_val:
                max_val = subset_sum
                max_idx = subset_idx

        query_idx = np.array(max_idx)
        return query_idx, X_pool[query_idx]


def mean_max_loss(classifier: BaseEstimator,
                  X_pool: modALinput,
                  n_instances: int = 1) -> Tuple[np.ndarray, modALinput]:
    """
    Mean Max Loss query strategy for SVM multilabel classification.

    For more details on this query strategy, see
    Li et al., Multilabel SVM active learning for image classification
    (http://dx.doi.org/10.1109/ICIP.2004.1421535)

    Args:
        classifier: The multilabel classifier for which the labels are to be queried. Should be an SVM model
            such as the ones from sklearn.svm. Although the function will execute for other models as well,
            the mathematical calculations in Li et al. work only for SVM-s.
        X: The pool of samples to query from.

    Returns:
        The index of the instance from X chosen to be labelled; the instance from X chosen to be labelled.
    """

    loss = _SVM_loss(classifier, X_pool)

    assert len(X_pool) >= n_instances, 'n_instances cannot be larger than len(X_pool)'

    if n_instances == 1:
        query_idx = np.argmax(loss)
        return query_idx, X_pool[query_idx]
    else:
        max_val = -np.inf
        max_idx = None
        for subset_idx in combinations(range(len(X_pool)), n_instances):
            subset_sum = loss[list(subset_idx)].sum()
            if subset_sum > max_val:
                max_val = subset_sum
                max_idx = subset_idx

        query_idx = np.array(max_idx)
        return query_idx, X_pool[query_idx]