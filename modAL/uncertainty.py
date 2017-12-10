"""
Uncertainty measures for the active learning models.
"""

import numpy as np
from scipy.stats import entropy


def classifier_uncertainty(classifier, data, **predict_proba_kwargs):
    # calculate uncertainty for each point provided
    classwise_uncertainty = classifier.predict_proba(data, **predict_proba_kwargs)

    # for each point, select the maximum uncertainty
    return 1 - np.max(classwise_uncertainty, axis=1)


def classifier_margin(classifier, data, **predict_proba_kwargs):
    classwise_uncertainty = classifier.predict_proba(data, **predict_proba_kwargs)

    if classwise_uncertainty.shape[1] == 1:
        return np.zeros(shape=(classwise_uncertainty.shape[0],))

    part = np.partition(-classwise_uncertainty, 1, axis=1)

    return -part[:, 0] + part[:, 1]


def classifier_entropy(classifier, data, **predict_proba_kwargs):
    classwise_uncertainty = classifier.predict_proba(data, **predict_proba_kwargs)

    entr = np.zeros(shape=(data.shape[0], ))

    for unc_idx, unc in enumerate(classwise_uncertainty):
        entr[unc_idx] = entropy(unc)

    return entr
