"""
Uncertainty measures that explicitly support batch-mode sampling for active learning models.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import pairwise_distances

from modAL.utils.combination import data_vstack
from modAL.models import BaseCommittee, BaseLearner
from modAL.uncertainty import classifier_uncertainty


def select_cold_start_instance(X: Union[np.ndarray, sp.csr_matrix],
                               metric: Union[str, Callable],
                               n_jobs: int) -> Union[np.ndarray, sp.csr_matrix]:
    """
    Define what to do if our batch-mode sampling doesn't have any labeled data -- a cold start.

    If our ranked batch sampling algorithm doesn't have any labeled data to determine
    similarity among the uncertainty set, this function finds the element with highest
    average similarity to cold-start the batch selection.

    TODO:
        - Figure out how to test this! E.g. how to create modAL model without training data.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    :param X:
        The set of unlabeled records.
    :type X:
        numpy.ndarray of shape (n_records, n_features)

    :param metric:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type metric:
        str or callable

    :param n_jobs:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type n_jobs:
        int

    :returns:
      - **X[best_instance_index]** *(numpy.ndarray or scipy.sparse.csr_matrix of shape (n_features, ))* -- Best instance
        for cold-start.
    """

    # Compute all pairwise distances in our unlabeled data and obtain the row-wise average for each of our records in X.
    average_distances = np.mean(pairwise_distances(X, metric=metric, n_jobs=n_jobs), axis=0)

    # Isolate and return our best instance for labeling as the record with the least average distance.
    best_coldstart_instance_index = np.argmin(average_distances)
    return X[best_coldstart_instance_index].reshape(1, -1)


def select_instance(
        X_training: Union[np.ndarray, sp.csr_matrix],
        X_pool: Union[np.ndarray, sp.csr_matrix],
        X_uncertainty: np.ndarray,
        mask: np.ndarray,
        metric: Union[str, Callable],
        n_jobs: int
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix], np.ndarray]:
    """
    Core iteration strategy for selecting another record from our unlabeled records.

    Given a set of labeled records (X_training) and unlabeled records (X_pool) with uncertainty
    scores (X_uncertainty), we'd like to identify the best instance in X_pool
    that best balances uncertainty and dissimilarity.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    TODO:
        - Add notebook for Active Learning bake-off (passive vs interactive vs batch vs ranked batch)

    :param X_training:
        Mix of both labeled and unlabeled records.
    :type X_training:
        numpy.ndarray of shape (D + batch_iteration, n_features)

    :param X_pool:
        Unlabeled records to be selected for labeling.
    :type X_pool:
        numpy.ndarray of shape (U - batch_iteration, n_features)

    :param X_uncertainty:
        Uncertainty scores for unlabeled records to be selected for labeling.
    :type X_uncertainty:
        numpy.ndarray of shape (U - batch_iteration,)

    :param mask:
        Mask to exclude previously selected instances from the pool
    :type mask:
        np.ndarray

    :param metric:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type metric:
        str or callable

    :param n_jobs:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type n_jobs:
        int

    :returns:
      - **best_instance_index** *int*
        -- Index of the best index from X chosen to be labelled.
      - **unlabeled_records[best_instance_index]** *(numpy.ndarray of shape (n_features, ))*
        -- a single record from our unlabeled set that is considered the most optimal
            incremental record for including in our query set.
    """

    # Extract the number of labeled and unlabeled records.
    n_labeled_records, _ = X_training.shape
    n_unlabeled, _ = X_pool[mask].shape

    # Determine our alpha parameter as |U| / (|U| + |D|). Note that because we
    # append to X_training and remove from X_pool within `ranked_batch`,
    # :alpha: is not fixed throughout our model's lifetime.
    alpha = n_unlabeled / (n_unlabeled + n_labeled_records)

    # Compute pairwise distance (and then similarity) scores from every unlabeled record
    # to every record in X_training. The result is an array of shape (n_samples, ).
    distance_scores = pairwise_distances(X_pool[mask], X_training, metric=metric, n_jobs=n_jobs).min(axis=1)

    similarity_scores = 1 / (1 + distance_scores)

    # Compute our final scores, which are a balance between how dissimilar a given record
    # is with the records in X_uncertainty and how uncertain we are about its class.
    scores = alpha * (1 - similarity_scores) + (1 - alpha) * X_uncertainty[mask]

    # Isolate and return our best instance for labeling as the one with the largest score.
    best_instance_index = np.argmax(scores)
    mask[best_instance_index] = 0
    return best_instance_index, X_pool[best_instance_index].reshape(1, -1), mask


def ranked_batch(classifier: Union[BaseLearner, BaseCommittee],
                 unlabeled: Union[np.ndarray, sp.csr_matrix],
                 uncertainty_scores: np.ndarray,
                 n_instances: int,
                 metric: Union[str, Callable],
                 n_jobs: int) -> np.ndarray:
    """
    Query our top :n_instances: to request for labeling.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    :param classifier:
        One of modAL's supported active learning models.
    :type classifier:
        modAL.models.BaseLearner or modAL.models.BaseCommittee

    :param unlabeled:
        Set of records to be considered for our active learning model.
    :type unlabeled:
        numpy.ndarray of shape: (n_samples, n_features).

    :param uncertainty_scores:
        Our classifier's predictions over the response variable.
    :type uncertainty_scores:
        numpy.ndarray of shape (n_samples, )

    :param n_instances:
        Limit on the number of records to query from our unlabeled set.
    :type n_instances:
        int

    :param metric:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type metric:
        str or callable

    :param n_jobs:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type n_jobs:
        int

    :returns:
      - **instance_index_ranking** *(numpy.ndarray of shape (n_instances, ))* -- The indices of
        the top n_instances ranked unlabelled samples.

    """
    # Make a local copy of our classifier's training data.
    if classifier.X_training is None:
        labeled = select_cold_start_instance(X=unlabeled, metric=metric, n_jobs=n_jobs)
    elif classifier.X_training.shape[0] > 0:
        labeled = classifier.X_training[:]

    # Define our record container and the maximum number of records to sample.
    instance_index_ranking = []
    ceiling = np.minimum(unlabeled.shape[0], n_instances)

    # mask for unlabeled initialized as transparent
    mask = np.ones(unlabeled.shape[0], np.bool)

    for _ in range(ceiling):

        # Receive the instance and corresponding index from our unlabeled copy that scores highest.
        instance_index, instance, mask = select_instance(X_training=labeled, X_pool=unlabeled,
                                                         X_uncertainty=uncertainty_scores, mask=mask,
                                                         metric=metric, n_jobs=n_jobs)

        # Add our instance we've considered for labeling to our labeled set. Although we don't
        # know it's label, we want further iterations to consider the newly-added instance so
        # that we don't query the same instance redundantly.
        labeled = data_vstack((labeled, instance))

        # Finally, append our instance's index to the bottom of our ranking.
        instance_index_ranking.append(instance_index)

    # Return numpy array, not a list.
    return np.array(instance_index_ranking)


def uncertainty_batch_sampling(classifier: Union[BaseLearner, BaseCommittee],
                               X: Union[np.ndarray, sp.csr_matrix],
                               n_instances: int = 20,
                               metric: Union[str, Callable] = 'euclidean',
                               n_jobs: int = 1,
                               **uncertainty_measure_kwargs: Optional[Dict]
                               ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    """
    Batch sampling query strategy. Selects the least sure instances for labelling.


    This strategy differs from `modAL.uncertainty.uncertainty_sampling` because, although
    it is supported, traditional active learning query strategies suffer from sub-optimal
    record selection when passing `n_instances` > 1. This sampling strategy extends the
    interactive uncertainty query sampling by allowing for batch-mode uncertainty query
    sampling. Furthermore, it also enforces a ranking -- that is, which records among the
    batch are most important for labeling?

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    :param classifier:
        One of modAL's supported active learning models.
    :type classifier:
        modAL.models.BaseLearner or modAL.models.BaseCommittee

    :param X:
        Set of records to be considered for our active learning model.
    :type X:
        numpy.ndarray or scipy.sparse.csr_matrix of shape (n_samples, n_features)

    :param n_instances:
        Number of records to return for labeling from `X`.
    :type n_instances:
        int

    :param metric:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type metric:
        str or callable

    :param n_jobs:
        This parameter is passed to sklearn.metrics.pairwise.pairwise_distances
    :type n_jobs:
        int

    :param uncertainty_measure_kwargs:
        Keyword arguments to be passed for the predict_proba method of the classifier.
    :type uncertainty_measure_kwargs:
        keyword arguments

    :returns:
      - **query_indices** *(numpy.ndarray of shape (n_instances, ))* -- Indices of the
        instances from X chosen to be labelled.
      - **X[query_indices]** *(numpy.ndarray or scipy.sparse.csr_matrix of shape (n_instances, n_features))*
        -- Records from X chosen to be labelled.
    """

    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    query_indices = ranked_batch(classifier, unlabeled=X, uncertainty_scores=uncertainty,
                                 n_instances=n_instances, metric=metric, n_jobs=n_jobs)
    return query_indices, X[query_indices]
