import numpy as np


def classifier_uncertainty(classifier, data):
    # calculate uncertainty for each point provided
    classwise_uncertainty = classifier.predict_proba(data)

    # for each point, select the maximum uncertainty
    return 1 - np.max(classwise_uncertainty, axis=1)


def classifier_margin(classifier, data):
    classwise_uncertainty = classifier.predict_proba(data)

    margins = np.zeros(shape=(classwise_uncertainty.shape[0], ))

    for point_idx, point_uncertainty_dist in enumerate(classwise_uncertainty):
        first_max = -np.inf
        second_max = -np.inf

        for val in point_uncertainty_dist:
            if val > first_max:
                second_max = first_max
                first_max = val

        if second_max != -np.inf:
            margins[point_idx] = first_max - second_max
        else:
            margins[point_idx] = 0.0

    return margins
