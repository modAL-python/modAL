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


def PI(mean, std, max_val, tradeoff):
    return ndtr((mean - max_val - tradeoff)/std)


def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff)*ndtr(z) + std*norm.pdf(z)


def UCB(mean, std, beta):
    return mean + beta*std


def optimizer_PI(optimizer, X, tradeoff=0):
    """
    Probability of improvement acquisition function for Bayesian optimization.

    Parameters
    ----------
    optimizer: modAL.models.BayesianEstimator object
        The BayesianEstimator object for which the utility is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The samples for which the probability of improvement is to be calculated.

    tradeoff: float
        Value controlling the tradeoff parameter.

    Returns
    -------
    pi: numpy.ndarray of shape (n_samples, )
        Probability of improvement utility score.
    """
    mean, std = optimizer.predict(X, return_std=True)
    std = std.reshape(-1, 1)

    return PI(mean, std, optimizer.y_max, tradeoff)


def optimizer_EI(optimizer, X, tradeoff=0):
    """
    Expected improvement acquisition function for Bayesian optimization.

    Parameters
    ----------
    optimizer: modAL.models.BayesianEstimator object
        The BayesianEstimator object for which the utility is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The samples for which the expected improvement is to be calculated.

    tradeoff: float
        Value controlling the tradeoff parameter.

    Returns
    -------
    ei: numpy.ndarray of shape (n_samples, )
        Expected improvement utility score.
    """
    mean, std = optimizer.predict(X, return_std=True)
    std = std.reshape(-1, 1)

    return EI(mean, std, optimizer.y_max, tradeoff)


def optimizer_UCB(optimizer, X, beta=1):
    """
    Upper confidence bound acquisition function for Bayesian optimization.

    Parameters
    ----------
    optimizer: modAL.models.BayesianEstimator object
        The BayesianEstimator object for which the utility is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The samples for which the upper confidence bound is to be calculated.

    beta: float
        Value controlling the beta parameter.

    Returns
    -------
    ucb: numpy.ndarray of shape (n_samples, )
        Upper confidence bound utility score.
    """
    mean, std = optimizer.predict(X, return_std=True)
    std = std.reshape(-1, 1)

    return UCB(mean, std, beta)


def max_PI(optimizer, X, tradeoff=0, n_instances=1):
    """
    Maximum PI query strategy. Selects the instance with highest probability of improvement.

    Parameters
    ----------
    optimizer: modAL.models.BayesianEstimator object
        The BayesianEstimator object for which the utility is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The samples for which the probability of improvement is to be calculated.

    tradeoff: float
        Value controlling the tradeoff parameter.

    n_instances: int
        Number of samples to be queried.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X chosen to be labelled.

    X[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X chosen to be labelled.
    """
    pi = optimizer_PI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(pi, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_EI(optimizer, X, tradeoff=0, n_instances=1):
    """
    Maximum EI query strategy. Selects the instance with highest expected improvement.

    Parameters
    ----------
    optimizer: modAL.models.BayesianEstimator object
        The BayesianEstimator object for which the utility is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The samples for which the expected improvement is to be calculated.

    tradeoff: float
        Value controlling the tradeoff parameter.

    n_instances: int
        Number of samples to be queried.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X chosen to be labelled.

    X[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X chosen to be labelled.
    """
    ei = optimizer_EI(optimizer, X, tradeoff=tradeoff)
    query_idx = multi_argmax(ei, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_UCB(optimizer, X, beta=1, n_instances=1):
    """
    Maximum UCB query strategy. Selects the instance with highest upper confidence
    bound.

    Parameters
    ----------
    optimizer: modAL.models.BayesianEstimator object
        The BayesianEstimator object for which the utility is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The samples for which the probability of improvement is to be calculated.

    beta: float
        Value controlling the beta parameter.

    n_instances: int
        Number of samples to be queried.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X chosen to be labelled.

    X[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X chosen to be labelled.
    """
    ucb = optimizer_UCB(optimizer, X, beta=beta)
    query_idx = multi_argmax(ucb, n_instances=n_instances)

    return query_idx, X[query_idx]
