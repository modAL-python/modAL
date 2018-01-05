"""
=====================
Disagreement sampling
=====================
-------------------------------------------------------------------------------------
Disagreement measures and disagreement based query strategies for the Committee model
-------------------------------------------------------------------------------------
"""

import numpy as np
from collections import Counter
from scipy.stats import entropy
from modAL.utils.selection import multi_argmax


def vote_entropy(committee, X, **predict_proba_kwargs):
    """
    Calculates the vote entropy for the Committee. First it computes the
    predictions of X for each learner in the Committee, then calculates
    the probability distribution of the votes. The entropy of this distribution
    is the vote entropy of the Committee, which is returned.

    Parameters
    ----------
    committee: modAL.models.Committee object
        The Committee instance for which the vote entropy is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The data for which the vote entropy is to be calculated.

    predict_proba_kwargs: keyword arguments
        Keyword arguments for the predict_proba method of the Committee.

    Returns
    -------
    entr: numpy.ndarray of shape (n_samples, )
        Vote entropy of the Committee for the samples in X.

    References
    ----------
    Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (3.1)
    """
    n_learners = len(committee)
    votes = committee.vote(X, **predict_proba_kwargs)
    p_vote = np.zeros(shape=(X.shape[0], len(committee.classes_)))
    entr = np.zeros(shape=(X.shape[0],))

    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)

        for class_idx, class_label in enumerate(committee.classes_):
            p_vote[vote_idx, class_idx] = vote_counter[class_label]/n_learners

        entr[vote_idx] = entropy(p_vote[vote_idx])

    return entr


def consensus_entropy(committee, X, **predict_proba_kwargs):
    """
    Calculates the consensus entropy for the Committee. First it computes the class
    probabilties of X for each learner in the Committee, then calculates the consensus
    probability distribution by averaging the individual class probabilities for each
    learner. The entropy of the consensus probability distribution is the vote entropy
    of the Committee, which is returned.

    Parameters
    ----------
    committee: modAL.models.Committee object
        The Committee instance for which the vote uncertainty entropy is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The data for which the vote uncertainty entropy is to be calculated.

    predict_proba_kwargs: keyword arguments
        Keyword arguments for the predict_proba method of the Committee.

    Returns
    -------
    entr: numpy.ndarray of shape (n_samples, )
        Vote uncertainty entropy of the Committee for the samples in X.

    References
    ----------
    Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (3.2)
    """
    proba = committee.predict_proba(X, **predict_proba_kwargs)
    entr = np.transpose(entropy(np.transpose(proba)))
    return entr


def KL_max_disagreement(committee, X, **predict_proba_kwargs):
    """
    Calculates the max disagreement for the Committee. First it computes the class probabilties
    of X for each learner in the Committee, then calculates the consensus probability
    distribution by averaging the individual class probabilities for each learner. Then each
    learner's class probabilities are compared to the consensus distribution in the sense of
    Kullback-Leibler divergence. The max disagreement for a given sample is the argmax of the
    KL divergences of the learners from the consensus probability.

    Parameters
    ----------
    committee: modAL.models.Committee object
        The Committee instance for which the max disagreement is to be calculated.

    X: numpy.ndarray of shape (n_samples, n_features)
        The data for which the max disagreement is to be calculated.

    predict_proba_kwargs: keyword arguments
        Keyword arguments for the predict_proba method of the Committee.

    Returns
    -------
    entr: numpy.ndarray of shape (n_samples, )
        Max disagreement of the Committee for the samples in X.

    References
    ----------
    Settles, Burr: Active Learning, (Morgan & Claypool Publishers), equation no. (3.3)
    """
    p_vote = committee.vote_proba(X, **predict_proba_kwargs)
    p_consensus = np.mean(p_vote, axis=1)

    learner_KL_div = np.zeros(shape=(len(X), len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_KL_div[:, learner_idx] = entropy(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    return np.max(learner_KL_div, axis=1)


def vote_entropy_sampling(committee, X, n_instances=1, **disagreement_measure_kwargs):
    """
    Vote entropy sampling strategy.

    Parameters
    ----------
    committee: Committee object
        The committee for which the labels are to be queried.

    X: numpy.ndarray of shape (n_samples, n_features)
        The pool of samples to query from.

    n_instances: int
        Number of samples to be queried.

    disagreement_measure_kwargs:
        Keyword arguments to be passed for the disagreement measure function.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X_pool chosen to be labelled.

    X_pool[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X_pool chosen to be labelled.
    """
    disagreement = vote_entropy(committee, X, **disagreement_measure_kwargs)
    query_idx = multi_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def consensus_entropy_sampling(committee, X, n_instances=1, **disagreement_measure_kwargs):
    """
    Consensus entropy sampling strategy.

    Parameters
    ----------
    committee: Committee object
        The committee for which the labels are to be queried.

    X: numpy.ndarray of shape (n_samples, n_features)
        The pool of samples to query from.

    n_instances: int
        Number of samples to be queried.

    disagreement_measure_kwargs:
        Keyword arguments to be passed for the disagreement measure function.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X_pool chosen to be labelled.

    X_pool[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X_pool chosen to be labelled.
    """
    disagreement = consensus_entropy(committee, X, **disagreement_measure_kwargs)
    query_idx = multi_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_disagreement_sampling(committee, X, n_instances=1, **disagreement_measure_kwargs):
    """
    Maximum disagreement sampling strategy.

    Parameters
    ----------
    committee: Committee object
        The committee for which the labels are to be queried.

    X: numpy.ndarray of shape (n_samples, n_features)
        The pool of samples to query from.

    n_instances: int
        Number of samples to be queried.

    disagreement_measure_kwargs:
        Keyword arguments to be passed for the disagreement measure function.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X_pool chosen to be labelled.

    X_pool[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X_pool chosen to be labelled.
    """
    disagreement = KL_max_disagreement(committee, X, **disagreement_measure_kwargs)
    query_idx = multi_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def max_std_sampling(regressor, X, n_instances=1, **predict_kwargs):
    """
    Regressor standard deviation sampling strategy.

    Parameters
    ----------
    X: numpy.ndarray of shape (n_samples, n_features)
        The pool of samples to query from.

    n_instances: int
        Number of samples to be queried.

    predict_kwargs:
        Keyword arguments to be passed for the predict method.

    Returns
    -------
    query_idx: numpy.ndarray of shape (n_instances, )
        The indices of the instances from X_pool chosen to be labelled.

    X_pool[query_idx]: numpy.ndarray of shape (n_instances, n_features)
        The instances from X_pool chosen to be labelled.
    """
    _, std = regressor.predict(X, return_std=True, **predict_kwargs)
    std = std.reshape(len(X), )
    query_idx = multi_argmax(std, n_instances=n_instances)
    return query_idx, X[query_idx]