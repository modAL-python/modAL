import numpy as np


def check_class_labels(*args):
    """
    Checks the known class labels for each classifier.
    :param args: sklearn classifiers
    :return: int or bool, False if the class labels are not the same for each predictor
                          number of classes if class labels are the same for each predictor
    """

    for classifier_idx in range(len(args) - 1):
        if not np.array_equal(args[classifier_idx].classes_, args[classifier_idx+1].classes_):
            return False

    return True


def check_class_proba(proba, known_labels, all_labels):
    """
    Checks the output of predict_proba and reshapes it if not all labels are present in the classifier
    :param proba:
    :param classes:
    :param all_classes:
    :return:
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
