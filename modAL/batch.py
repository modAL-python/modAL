"""
    Uncertainty measures that explicitly support batch-mode sampling for active learning models.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform

from modAL.density import euclidean_similarity
from modAL.models import BaseCommittee, BaseLearner
from modAL.uncertainty import classifier_uncertainty


def select_cold_start_instance(X: np.ndarray, similarity_fn: Callable = euclidean_similarity) -> np.ndarray:
    """Define what to do if our batch-mode sampling doesn't have any labeled data -- a cold start.

    If our ranked batch sampling algorithm doesn't have any labeled data to determine
    similarity among the uncertainty set,

    TODO:
        - Figure out how to test this! E.g. how to create modAL model without training data.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    :param X: the set of unlabeled records. numpy.ndarray of shape (n_records, n_features).
    :param similarity_fn: a function that takes two N-length vectors and returns a similarity in
        range [0, 1].
    :return: X[best_instance_index]
    """

    # Compute all pairwise similarities in our unlabeled data.
    pairwise_similarities = squareform(pdist(X, similarity_fn))

    # Obtain the row-wise average for each of our records in X.
    average_similarity = np.mean(pairwise_similarities, axis=0)

    # Isolate and return our best instance for labeling as the
    # record with the greatest average similarity.
    best_coldstart_instance_index = np.argmax(average_similarity)
    return X[best_coldstart_instance_index]


def select_instance(X_training: np.ndarray,
                    X_uncertainty: np.ndarray,
                    similarity_fn: Callable = euclidean_similarity) -> np.ndarray:
    """Core iteration strategy for selecting another record from our unlabeled records.

    Given a set of labeled records (X_training) and unlabeled records with uncertainty
    scores (X_uncertainty), we'd like to identify the best instance in X_uncertainty
    that best balances uncertainty and dissimilarity.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    TODO:
        - Add notebook for Active Learning bake-off (passive vs interactive vs batch vs ranked batch)

    :param X_training: mix of both labeled and unlabeled records.
        Array of shape (D + batch_iteration, n_features).
    :param X_uncertainty: unlabeled records to be selected for labeling.
        Array of shape (U - batch_iteration, n_features).
    :param similarity_fn: a function that takes two N-length vectors and returns a similarity in
        range [0, 1]. Note: any distance function can be a similarity function as 1 / (1 + d)
        where d is the distance.
    :return: unlabeled_records[best_instance_index]: a single record from our unlabeled set that
        is considered the most optimal incremental record for including in our query set.
        numpy.ndarray of shape (n_features, ).
    """

    def max_vector_matrix_distance(arr: np.ndarray, pool: np.ndarray) -> np.float:
        """Compute the maximum of pairwise similarity between a flat array and a matrix.

        :param arr: single vector of shape (n_features, ).
        :param pool: matrix of shape (n_features, n_records).
        :return: numeric corresponding to the maximum similarity between our vector and matrix.
        """
        return np.max([similarity_fn(arr, x_i) for x_i in pool])

    # Determine our alpha parameter as |U| / (|U| + |D|). Note that because we
    # append to X_training and remove from X_uncertainty within `ranked_batch`,
    # :alpha: is not fixed throughout our model's lifetime.
    n_labeled, n_unlabeled = X_training.shape[0], X_uncertainty.shape[0]
    alpha = n_unlabeled / (n_unlabeled + n_labeled)

    # Isolate our original unlabeled records from their predicted class uncertainty.
    unlabeled_records, uncertainty_scores = X_uncertainty[:, :-1], X_uncertainty[:, -1]

    # Compute pairwise similarity scores from every unlabeled record in unlabeled_records
    # to every record in X_training. The result is an array of shape (n_samples, ).
    similarity_scores = np.apply_along_axis(max_vector_matrix_distance, arr=unlabeled_records.T, pool=X_training, axis=0)

    # Compute our final scores, which are a balance between how dissimilar a given record
    # is with the records in X_uncertainty and how uncertain we are about its class.
    scores = alpha * (1 - similarity_scores) + (1 - alpha) * uncertainty_scores

    # Isolate and return our best instance for labeling as the one with the largest score.
    best_instance_index = np.argmax(scores)
    return unlabeled_records[best_instance_index]


def ranked_batch(classifier: Union[BaseLearner, BaseCommittee],
                 unlabeled: np.ndarray,
                 uncertainty_scores: np.ndarray,
                 n_instances: int) -> np.ndarray:
    """Query our top :n_instances: to request for labeling.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    :param classifier: one of modAL's supported active learning models.
    :param unlabeled: set of records to be considered for our active learning model.
        Shape: (n_samples, n_features).
    :param uncertainty_scores: our classifier's predictions over the response variable.
        Shape (n_samples, ).
    :param n_instances: limit on the number of records to query from our unlabeled set.
    :return:
    """

    # Make a local copy of our classifier's training data.
    n_training_records = classifier.X_training.shape[0]
    labeled = np.copy(classifier.X_training) if n_training_records > 0 else select_cold_start_instance(unlabeled)

    # Add uncertainty scores to our unlabeled data, and keep a copy of our unlabeled data.
    unlabeled_uncertainty = np.concatenate((unlabeled, np.expand_dims(uncertainty_scores, axis=1)), axis=1)
    unlabeled_uncertainty_copy = np.copy(unlabeled_uncertainty)

    # Define our record container and the maximum number of records to sample.
    instance_index_ranking = []
    ceiling = np.minimum(unlabeled.shape[0], n_instances)

    # TODO (dataframing) is there a better way to do this? Inherently sequential.
    for _ in range(ceiling):

        # Select the instance from our unlabeled copy that scores highest.
        raw_instance = select_instance(X_training=labeled, X_uncertainty=unlabeled_uncertainty_copy)
        instance = np.expand_dims(raw_instance, axis=1)

        # Find our record's index in both the original unlabeled and our uncertainty copy.
        instance_index_original = np.where(np.all(unlabeled == raw_instance, axis=1))[0][0]
        instance_index_copy = np.where(np.all(unlabeled_uncertainty_copy[:, :-1] == instance.T, axis=1))[0][0]

        # Add our instance we've considered for labeling to our labeled set. Although we don't
        # know it's label, we want further iterations to consider the newly-added instance so
        # that we don't query the same instance redundantly.
        labeled = np.concatenate((labeled, instance.T), axis=0)

        # Remove our instance from the unlabeled set and append it to our list of records to label.
        unlabeled_uncertainty_copy = np.delete(unlabeled_uncertainty_copy, instance_index_copy, axis=0)
        instance_index_ranking.append(instance_index_original)

    # Return numpy array, not a list.
    return np.array(instance_index_ranking)


def uncertainty_batch_sampling(classifier: Union[BaseLearner, BaseCommittee],
                               X: np.ndarray,
                               n_instances: int = 20,
                               **uncertainty_measure_kwargs: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Batch sampling query strategy. Selects the least sure instances for labelling.

    This strategy differs from `modAL.uncertainty.uncertainty_sampling` because, although
    it is supported, traditional active learning query strategies suffer from sub-optimal
    record selection when passing `n_instances` > 1. This sampling strategy extends the
    interactive uncertainty query sampling by allowing for batch-mode uncertainty query
    sampling. Furthermore, it also enforces a ranking -- that is, which records among the
    batch are most important for labeling?

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    :param classifier: one of modAL's supported active learning models.
    :param X: set of records to be considered for our active learning model. Shape: (n_samples, n_features).
    :param n_instances: indicator for the number of records to return for labeling from `X`.
    :param uncertainty_measure_kwargs: keyword arguments to be passed for the predict_proba method of the classifier.

    :returns tuple (query_indices, X[query_indices)
        query_indices: indices of the instances from X chosen to be labelled. numpy.ndarray of shape (n_instances, ).
        X[query_indices]: records from X chosen to be labelled. numpy.ndarray of shape (n_instances, n_features).
    """

    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    query_indices = ranked_batch(classifier, unlabeled=X, uncertainty_scores=uncertainty, n_instances=n_instances)
    return query_indices, X[query_indices]
