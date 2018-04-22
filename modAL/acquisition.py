"""
=====================
Acquisition functions
=====================
-----------------------------------------------
Acquisition functions for Bayesian optimization
-----------------------------------------------
"""

from scipy.special import ndtr
from modAL.utils.selection import multi_argmax


def PI(optimizer, X, tradeoff=0):
    mean, std = optimizer.predict(X, return_std=True)
    return ndtr((mean - optimizer.max_val - tradeoff)/std)


def max_PI(optimizer, X, tradeoff=0, n_instances=1):
    pi = PI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(pi, n_instances=n_instances)

    return query_idx, X[query_idx]