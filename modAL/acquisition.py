"""
Acquisition functions for Bayesian optimization.
"""
from typing import Tuple

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.exceptions import NotFittedError

from modAL.utils.selection import multi_argmax
from modAL.utils.data import modALinput
from modAL.models.base import BaseLearner


def PI(mean, std, max_val, tradeoff):
    return ndtr((mean - max_val - tradeoff)/std)


def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff)*ndtr(z) + std*norm.pdf(z)


def UCB(mean, std, beta):
    return mean + beta*std


"""
---------------------
Acquisition functions
---------------------
"""


def optimizer_PI(optimizer: BaseLearner, X: modALinput, tradeoff: float = 0) -> np.ndarray:
    """
    Probability of improvement acquisition function for Bayesian optimization.

    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the probability of improvement is to be calculated.
        tradeoff: Value controlling the tradeoff parameter.

    Returns:
        Probability of improvement utility score.
    """
    try:
        mean, std = optimizer.predict(X, return_std=True)
        std = std.reshape(-1, 1)
    except NotFittedError:
        mean, std = np.zeros(shape=(X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))

    return PI(mean, std, optimizer.y_max, tradeoff)


def optimizer_EI(optimizer: BaseLearner, X: modALinput, tradeoff: float = 0) -> np.ndarray:
    """
    Expected improvement acquisition function for Bayesian optimization.

    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the expected improvement is to be calculated.
        tradeoff: Value controlling the tradeoff parameter.

    Returns:
        Expected improvement utility score.
    """
    try:
        mean, std = optimizer.predict(X, return_std=True)
        std = std.reshape(-1, 1)
    except NotFittedError:
        mean, std = np.zeros(shape=(X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))

    return EI(mean, std, optimizer.y_max, tradeoff)


def optimizer_UCB(optimizer: BaseLearner, X: modALinput, beta: float = 1) -> np.ndarray:
    """
    Upper confidence bound acquisition function for Bayesian optimization.

    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the upper confidence bound is to be calculated.
        beta: Value controlling the beta parameter.

    Returns:
        Upper confidence bound utility score.
    """
    try:
        mean, std = optimizer.predict(X, return_std=True)
        std = std.reshape(-1, 1)
    except NotFittedError:
        mean, std = np.zeros(shape=(X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))

    return UCB(mean, std, beta)


"""
--------------------------------------------
Query strategies using acquisition functions
--------------------------------------------
"""


def max_PI(optimizer: BaseLearner, X: modALinput, tradeoff: float = 0,
           n_instances: int = 1) -> Tuple[np.ndarray, modALinput]:
    """
    Maximum PI query strategy. Selects the instance with highest probability of improvement.

    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the probability of improvement is to be calculated.
        tradeoff: Value controlling the tradeoff parameter.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled; the instances from X chosen to be labelled.
    """
    pi = optimizer_PI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(pi, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_EI(optimizer: BaseLearner, X: modALinput, tradeoff: float = 0,
           n_instances: int = 1) -> Tuple[np.ndarray, modALinput]:
    """
    Maximum EI query strategy. Selects the instance with highest expected improvement.

    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the expected improvement is to be calculated.
        tradeoff: Value controlling the tradeoff parameter.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled; the instances from X chosen to be labelled.
    """
    ei = optimizer_EI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(ei, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_UCB(optimizer: BaseLearner, X: modALinput, beta: float = 1,
            n_instances: int = 1) -> Tuple[np.ndarray, modALinput]:
    """
    Maximum UCB query strategy. Selects the instance with highest upper confidence bound.

    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the maximum upper confidence bound is to be calculated.
        beta: Value controlling the beta parameter.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled; the instances from X chosen to be labelled.
    """
    ucb = optimizer_UCB(optimizer, X, beta=beta)
    query_idx = multi_argmax(ucb, n_instances=n_instances)

    return query_idx, X[query_idx]
