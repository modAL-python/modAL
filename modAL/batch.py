from collections import deque
from typing import Dict, Optional, Callable, Union

import numpy as np

from modAL.density import euclidean_similarity
from modAL.models import BaseLearner, BaseCommittee
from modAL.uncertainty import classifier_uncertainty


def select_instance(X_training: np.ndarray,
                    X_with_uncertainty: np.ndarray,
                    similarity_fn: Callable = euclidean_similarity):
    """
    TODO

    :param X_training:
    :param X:
    :param X_uncertainty:
    :param similarity_fn:
    :return:
    """

    n_labeled, n_unlabeled = X_training.shape[0], X_with_uncertainty.shape[0]
    alpha = n_unlabeled / (n_unlabeled + n_labeled)

    def vector_matrix_distance(arr: np.ndarray, pool: np.ndarray):
        return np.max([similarity_fn(arr, x_i) for x_i in pool])

    unlabeled_records, classifier_uncertainty_scores = X_with_uncertainty[:, :-1], X_with_uncertainty[:, -1]
    similarity_scores = np.apply_along_axis(vector_matrix_distance, arr=unlabeled_records.T, pool=X_training, axis=0)

    scores = alpha * (1 - similarity_scores) + (1 - alpha) * classifier_uncertainty_scores

    best_instance_index = np.argmax(scores)
    best_instance = unlabeled_records[best_instance_index]

    return best_instance


def ranked_batch(classifier: Union[BaseLearner, BaseCommittee],
                 unlabeled: np.ndarray,
                 uncertainty_scores: np.ndarray,
                 n_instances: int) -> np.ndarray:
    """
    TODO

    :param X_training:
    :param X:
    :param X_uncertainty:
    :param n_instances:
    :return:
    """

    labeled = np.copy(classifier.X_training)

    unlabeled_uncertainty = np.concatenate((unlabeled, np.expand_dims(uncertainty_scores, axis=1)), axis=1)
    unlabeled_uncertainty_copy = np.copy(unlabeled_uncertainty)

    instance_index_ranking = deque()
    ceiling = np.minimum(unlabeled.shape[0], n_instances)

    # TODO (dataframing) there must be a better way...maybe?
    for _ in range(ceiling):

        raw_instance = select_instance(X_training=labeled, X_with_uncertainty=unlabeled_uncertainty_copy)
        instance = np.expand_dims(raw_instance, axis=1)

        instance_index_original = np.where(np.all(unlabeled == raw_instance, axis=1))[0][0]
        instance_index_copy = np.where(np.all(unlabeled_uncertainty_copy[:, :-1] == instance.T, axis=1))[0][0]

        labeled = np.concatenate((labeled, instance.T), axis=0)
        unlabeled_uncertainty_copy = np.delete(unlabeled_uncertainty_copy, instance_index_copy, axis=0)

        instance_index_ranking.append(instance_index_original)

    return np.array(instance_index_ranking)


def uncertainty_batch_sampling(classifier: Union[BaseLearner, BaseCommittee],
                               X: np.ndarray,
                               n_instances: int = 20,
                               **uncertainty_measure_kwargs: Optional[Dict]):
    """
    TODO

    :param classifier:
    :param X:
    :param n_instances:
    :param uncertainty_measure_kwargs:
    :return:
    """

    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    query_indices = ranked_batch(classifier, unlabeled=X, uncertainty_scores=uncertainty, n_instances=n_instances)
    return query_indices, X[query_indices]
