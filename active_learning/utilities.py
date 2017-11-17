import numpy as np


def classifier_uncertainty(classifier, data):
    # calculate uncertainty for each point provided
    classwise_uncertainty = classifier.predict_proba(data)

    # for each point, select the maximum uncertainty
    return 1 - np.max(classwise_uncertainty, axis=1)

