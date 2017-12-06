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
