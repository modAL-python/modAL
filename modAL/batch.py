"""
Uncertainty measures that explicitly support batch-mode sampling for active learning models.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min

from modAL.utils.data import data_vstack, modALinput
from modAL.models.base import BaseCommittee, BaseLearner
from modAL.uncertainty import classifier_uncertainty


def select_cold_start_instance(X: modALinput,
                               metric: Union[str, Callable],
                               n_jobs: Union[int, None]) -> Tuple[int, modALinput]:
    """
    Define what to do if our batch-mode sampling doesn't have any labeled data -- a cold start.

    If our ranked batch sampling algorithm doesn't have any labeled data to determine similarity among the uncertainty
    set, this function finds the element with highest average similarity to cold-start the batch selection.

    TODO:
        - Figure out how to test this! E.g. how to create modAL model without training data.
        - Think of optimizing pairwise_distance call for large matrix.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    Args:
        X: The set of unlabeled records.
        metric: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
        n_jobs: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.

    Returns:
        Index of the best cold-start instance from `X` chosen to be labelled; record of the best cold-start instance
        from `X` chosen to be labelled.
    """
    # Compute all pairwise distances in our unlabeled data and obtain the row-wise average for each of our records in X.
    n_jobs = n_jobs if n_jobs else 1
    average_distances = np.mean(pairwise_distances(X, metric=metric, n_jobs=n_jobs), axis=0)

    # Isolate and return our best instance for labeling as the record with the least average distance.
    best_coldstart_instance_index = np.argmin(average_distances)
    return best_coldstart_instance_index, X[best_coldstart_instance_index].reshape(1, -1)


def select_instance(
        X_similarity_scores: np.ndarray,
        X_uncertainty: np.ndarray,
        mask: np.ndarray,
        alpha: float,
) -> Tuple[np.ndarray, modALinput, np.ndarray]:
    """
    Core iteration strategy for selecting another record from our unlabeled records.

    Given a set of labeled records (X_training) and unlabeled records (X_pool) with uncertainty scores (X_uncertainty),
    we'd like to identify the best instance in X_pool that best balances uncertainty and dissimilarity.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    TODO:
        - Add notebook for Active Learning bake-off (passive vs interactive vs batch vs ranked batch)

    Args:
        X_similarity_scores: Similarity between unlabeled (or already picked) records and labeled.
        X_uncertainty: Uncertainty scores for unlabeled records to be selected for labeling.
        mask: Mask to exclude previously selected instances from the pool.

    Returns:
        Index of the best index from X chosen to be labelled; a single record from our unlabeled set that is considered
        the most optimal incremental record for including in our query set.
    """

    # Compute our final scores, which are a balance between how dissimilar a given record
    # is with the records in X_uncertainty and how uncertain we are about its class.
    scores = alpha * (1 - X_similarity_scores[mask]) + (1 - alpha) * X_uncertainty[mask]

    # Isolate and return our best instance for labeling as the one with the largest score.
    best_instance_index_in_unlabeled = np.argmax(scores)
    return best_instance_index_in_unlabeled


def ranked_batch(classifier: Union[BaseLearner, BaseCommittee],
                 unlabeled: modALinput,
                 uncertainty_scores: np.ndarray,
                 n_instances: int,
                 metric: Union[str, Callable],
                 n_jobs: Union[int, None]) -> np.ndarray:
    """
    Query our top :n_instances: to request for labeling.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    Args:
        classifier: One of modAL's supported active learning models.
        unlabeled: Set of records to be considered for our active learning model.
        uncertainty_scores: Our classifier's predictions over the response variable.
        n_instances: Limit on the number of records to query from our unlabeled set.
        metric: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
        n_jobs: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.

    Returns:
        The indices of the top n_instances ranked unlabelled samples.
    """
    # Make a local copy of our classifier's training data.
    # Define our record container and record the best cold start instance in the case of cold start.
    if classifier.X_training is None:
        best_coldstart_instance_index, labeled = select_cold_start_instance(X=unlabeled, metric=metric, n_jobs=n_jobs)
        instance_index_ranking = [best_coldstart_instance_index]
    elif classifier.X_training.shape[0] > 0:
        labeled = classifier.X_training[:]
        instance_index_ranking = []
    
    # Extract the number of labeled and unlabeled records.
    n_labeled_records, *rest = labeled.shape
    n_unlabeled, *rest = unlabeled.shape

    # The maximum number of records to sample.
    ceiling = np.minimum(n_unlabeled, n_instances) - len(instance_index_ranking)

    # mask for unlabeled initialized as transparent
    mask = np.ones(n_unlabeled, np.bool)

    # Compute pairwise distance (and then similarity) scores from every unlabeled record
    # to every record in X_training. The result is an array of shape (n_samples, ).

    if n_jobs == 1 or n_jobs is None:
        _, distance_scores = pairwise_distances_argmin_min(unlabeled,
                                                           labeled,
                                                           metric=metric)
    else:
        distance_scores = pairwise_distances(unlabeled,
                                             labeled,
                                             metric=metric, n_jobs=n_jobs).min(axis=1)

    similarity_scores = 1 / (1 + distance_scores)

    for _ in range(ceiling):
        # Determine our alpha parameter as |U| / (|U| + |D|). Note that because we
        # append to X_training and remove from X_pool within `ranked_batch`,
        # :alpha: is not fixed throughout our model's lifetime.
        alpha = n_unlabeled / (n_unlabeled + n_labeled_records)

        # Receive the instance and corresponding index from our unlabeled copy that scores highest.
        instance_index = select_instance(X_similarity_scores=similarity_scores,
                                         X_uncertainty=uncertainty_scores, mask=mask, alpha=alpha)
        
        # Consider the newly selected sample as labeled
        instance_vector = unlabeled[mask][[instance_index]]
        mask[instance_index] = 0
        n_unlabeled -= 1
        n_labeled_records += 1

        # Compute distance to labeled
        distance_to_labeled = pairwise_distances(unlabeled,
                                                 instance_vector,
                                                 metric=metric)[:, 0]

        # Add our instance we've considered for labeling to our labeled set. Although we don't
        # know it's label, we want further iterations to consider the newly-added instance so
        # that we don't query the same instance redundantly.
        similarity_scores = np.max([similarity_scores, 1 / (1 + distance_to_labeled)], axis=0)

        # Finally, append our instance's index to the bottom of our ranking.
        instance_index_ranking.append(instance_index)

    # Return numpy array, not a list.
    return np.array(instance_index_ranking)


def uncertainty_batch_sampling(classifier: Union[BaseLearner, BaseCommittee],
                               X: Union[np.ndarray, sp.csr_matrix],
                               n_instances: int = 20,
                               metric: Union[str, Callable] = 'euclidean',
                               n_jobs: Optional[int] = None,
                               **uncertainty_measure_kwargs
                               ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    """
    Batch sampling query strategy. Selects the least sure instances for labelling.

    This strategy differs from :func:`~modAL.uncertainty.uncertainty_sampling` because, although it is supported,
    traditional active learning query strategies suffer from sub-optimal record selection when passing
    `n_instances` > 1. This sampling strategy extends the interactive uncertainty query sampling by allowing for
    batch-mode uncertainty query sampling. Furthermore, it also enforces a ranking -- that is, which records among the
    batch are most important for labeling?

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    Args:
        classifier: One of modAL's supported active learning models.
        X: Set of records to be considered for our active learning model.
        n_instances: Number of records to return for labeling from `X`.
        metric: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`
        n_jobs: If not set, :func:`~sklearn.metrics.pairwise.pairwise_distances_argmin_min` is used for calculation of
            distances between samples. Otherwise it is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Indices of the instances from `X` chosen to be labelled; records from `X` chosen to be labelled.
    """
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    query_indices = ranked_batch(classifier, unlabeled=X, uncertainty_scores=uncertainty,
                                 n_instances=n_instances, metric=metric, n_jobs=n_jobs)
    return query_indices, X[query_indices]
