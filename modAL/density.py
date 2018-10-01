"""
Measures for estimating the information density of a given sample.
"""
from typing import Callable, Union

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import pairwise_distances

from modAL.utils.data import modALinput


def similarize_distance(distance_measure: Callable) -> Callable:
    """
    Takes a distance measure and converts it into a information_density measure.

    Args:
        distance_measure: The distance measure to be converted into information_density measure.

    Returns:
        The information_density measure obtained from the given distance measure.
    """
    def sim(*args, **kwargs):
        return 1/(1 + distance_measure(*args, **kwargs))

    return sim


cosine_similarity = similarize_distance(cosine)
euclidean_similarity = similarize_distance(euclidean)


def information_density(X: modALinput, metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    """
    Calculates the information density metric of the given data using the given metric.

    Args:
        X: The data for which the information density is to be calculated.
        metric: The metric to be used. Should take two 1d numpy.ndarrays for argument.

    Todo:
        Should work with all possible modALinput.
        Perhaps refactor the module to use some stuff from sklearn.metrics.pairwise

    Returns:
        The information density for each sample.
    """
    # inf_density = np.zeros(shape=(X.shape[0],))
    # for X_idx, X_inst in enumerate(X):
    #     inf_density[X_idx] = sum(similarity_measure(X_inst, X_j) for X_j in X)
    #
    # return inf_density/X.shape[0]

    similarity_mtx = 1/(1+pairwise_distances(X, X, metric=metric))

    return similarity_mtx.mean(axis=1)
