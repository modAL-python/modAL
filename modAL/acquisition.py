"""
=====================
Acquisition functions
=====================
-----------------------------------------------
Acquisition functions for Bayesian optimization
-----------------------------------------------
"""

from scipy.stats import norm
from scipy.special import ndtr
from modAL.utils.selection import multi_argmax


def PI(optimizer, X, tradeoff=0):
    mean, std = optimizer.predict(X, return_std=True)

    return ndtr((mean - optimizer.max_val - tradeoff)/std)


def EI(optimizer, X, tradeoff=0):
    mean, std = optimizer.predict(X, return_std=True)
    z = (mean - optimizer.max_val - tradeoff)/std

    return (mean - optimizer.max_val - tradeoff)*ndtr(z) + std*norm.pdf(z)


def max_PI(optimizer, X, tradeoff=0, n_instances=1):
    pi = PI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(pi, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_EI(optimizer, X, tradeoff=0, n_instances=1):
    ei = EI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(ei, n_instances=n_instances)

    return query_idx, X[query_idx]
