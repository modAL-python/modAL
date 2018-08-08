"""
Measures for estimating the information density of a given sample.
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean


def similarize_distance(distance_measure):
    """
    Takes a distance measure and converts it into a information_density measure.

    :param distance_measure:
        The distance measure to be converted into information_density measure.
    :type distance_measure:
        function

    :returns:
      - **sim** *(function)* --
        The information_density measure obtained from the given disance measure.
    """
    def sim(*args, **kwargs):
        return 1/(1 + distance_measure(*args, **kwargs))

    return sim


cosine_similarity = similarize_distance(cosine)
euclidean_similarity = similarize_distance(euclidean)


def information_density(X, similarity_measure=cosine_similarity):
    """
    Calculates the information density metric of the given data using the similarity
    measure given.

    :param X:
        The data for which the information density is to be calculated.
    :type X:
        numpy.ndarray of shape (n_samples, n_features)

    :param similarity_measure:
        The similarity measure to be used. Should take two 1d numpy.ndarrays for argument.
    :type similarity_measure:
        function

    :returns:
      - **inf_density** *(numpy.ndarray of shape (n_samples, ))* --
        The information density for each sample.

    """
    inf_density = np.zeros(shape=(len(X),))
    for X_idx, X_inst in enumerate(X):
        inf_density[X_idx] = sum(similarity_measure(X_inst, X_j) for X_j in X)

    return inf_density/len(X)
