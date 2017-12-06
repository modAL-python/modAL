def check_class_labels(*args):
    """
    Checks the known class labels for each classifier.
    :param args: sklearn classifiers
    :return: bool, True if the class labels are the same for each predictor, false otherwise
    """

    for classifier_idx in range(len(args) - 1):
        if not np.all(args[classifier_idx].classes_ == args[classifier_idx+1].classes_):
            return False

    return True