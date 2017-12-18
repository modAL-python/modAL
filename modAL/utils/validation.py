import numpy as np


def check_class_labels(*args):
    """
    Checks the known class labels for each classifier. Returns True if all classifier
    knows the same labels and returns False if not.

    Parameters
    ----------
    *args: sklearn classifier objects
        Classifier objects to check the known class labels.

    Returns
    -------
    bool
        True, if class labels match for all classifiers,
        False otherwise.
    """

    for classifier_idx in range(len(args) - 1):
        if not np.array_equal(args[classifier_idx].classes_, args[classifier_idx+1].classes_):
            return False

    return True


def check_class_proba(proba, known_labels, all_labels):
    """
    Checks the class probabilities and reshapes it if not all labels are present
    in the classifier.

    Parameters
    ----------
    proba: numpy.ndarray of shape (n_samples, n_known_classes)
        The class probabilities of a classifier.

    known_labels:
        The class labels known by the classifier.

    all_labels:
        All class labels.

    Returns
    -------
    aug_proba: numpy.ndarray of shape (n_samples, n_classes)
        Class probabilities augmented such that the probability of all classes
        is present. If the classifier is unaware of a particular class, all
        probabilities are zero.

    """

    label_idx_map = -np.ones(len(all_labels), dtype='int')

    for known_label_idx, known_label in enumerate(known_labels):
        # finds the position of label in all_labels
        for label_idx, label in enumerate(all_labels):
            if np.array_equal(label, known_label):
                label_idx_map[label_idx] = known_label_idx
                break

    aug_proba = np.hstack((proba, np.zeros(shape=(proba.shape[0], 1))))
    return aug_proba[:, label_idx_map]
