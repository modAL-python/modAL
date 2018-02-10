"""
============================
Information density measures
============================
-----------------------------------------------------------------
Measures for estimating the information density of a given sample
-----------------------------------------------------------------
"""

import numpy as np
from scipy.spatial.distance import cosine


def similarize_distance(distance_measure):
    """
    Takes a distance measure and converts it into a information_density measure.

    Parameters
    ----------
    distance_measure: function
        The distance measure to be converted into information_density measure.

    Returns
    -------
    sim: function
        The information_density measure obtained from the given disance measure.
    """
    def sim(*args, **kwargs):
        return 1/(1 + distance_measure(*args, **kwargs))

    return sim


cosine_similarity = similarize_distance(cosine)


def information_density(X, similarity_measure=cosine_similarity):
    """
    Calculates the information density metric of the given data using the similarity
    measure given.

    Parameters
    ----------
    X: numpy.ndarray of shape (n_samples, n_features)
        The data for which the information density is to be calculated.

    similarity_measure: function
        The similarity measure to be used. Should take two 1d numpy.ndarrays for argument.

    Returns
    -------
    inf_density: numpy.ndarray of shape (n_samples, )
        The information density for each sample.

    """
    inf_density = np.zeros(shape=(len(X),))
    for X_idx, X_inst in enumerate(X):
        inf_density[X_idx] = sum(similarity_measure(X_inst, X_j) for X_j in X)

    return inf_density/len(X)
