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

    def sim(*args, **kwargs):
        return 1/(1 + distance_measure(*args, **kwargs))

    return sim


def similarity(X_pool, similarity_measure=similarize_distance(cosine)):
    sim = np.zeros(shape=(len(X_pool), ))
    for X_idx, X in enumerate(X_pool):
        sim[X_idx] = sum(similarity_measure(X, X_j) for X_j in X_pool)

    return sim/len(X_pool)
