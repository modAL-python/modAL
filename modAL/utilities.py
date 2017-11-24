"""
Utility functions for the active learning model.
"""

import numpy as np
import bottleneck as bn


def classifier_uncertainty(classifier, data):
    # calculate uncertainty for each point provided
    classwise_uncertainty = classifier.predict_proba(data)

    # for each point, select the maximum uncertainty
    return 1 - np.max(classwise_uncertainty, axis=1)


def classifier_margin(classifier, data):
    classwise_uncertainty = classifier.predict_proba(data)

    if classwise_uncertainty.shape[1] == 1:
        return np.zeros(shape=(classwise_uncertainty.shape[0],))

    part = bn.partition(-classwise_uncertainty, 1, axis=1)

    return -part[:, 0] + part[:, 1]
