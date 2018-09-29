from typing import Sequence

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator


def check_class_labels(*args: BaseEstimator) -> bool:
    """
    Checks the known class labels for each classifier.

    Args:
        *args: Classifier objects to check the known class labels.

    Returns:
        True, if class labels match for all classifiers, False otherwise.
    """
    try:
        classes_ = [estimator.classes_ for estimator in args]
    except AttributeError:
        raise NotFittedError('Not all estimators are fitted. Fit all estimators before using this method.')

    for classifier_idx in range(len(args) - 1):
        if not np.array_equal(classes_[classifier_idx], classes_[classifier_idx+1]):
            return False

    return True


def check_class_proba(proba: np.ndarray, known_labels: Sequence, all_labels: Sequence) -> np.ndarray:
    """
    Checks the class probabilities and reshapes it if not all labels are present in the classifier.

    Args:
        proba: The class probabilities of a classifier.
        known_labels: The class labels known by the classifier.
        all_labels: All class labels.

    Returns:
        Class probabilities augmented such that the probability of all classes is present. If the classifier is unaware
        of a particular class, all probabilities are zero.
    """
    # TODO: rewrite this function using numpy.insert

    label_idx_map = -np.ones(len(all_labels), dtype='int')

    for known_label_idx, known_label in enumerate(known_labels):
        # finds the position of label in all_labels
        for label_idx, label in enumerate(all_labels):
            if np.array_equal(label, known_label):
                label_idx_map[label_idx] = known_label_idx
                break

    aug_proba = np.hstack((proba, np.zeros(shape=(proba.shape[0], 1))))
    return aug_proba[:, label_idx_map]
