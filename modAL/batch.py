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

    :param similarity_fn:
        A function that takes two N-length vectors and returns a similarity in range [0, 1].
    :type similarity_fn:
        Callable

    :returns:
      - **X[best_instance_index]** *(numpy.ndarray of shape (n_features, ))* -- Best instance for cold-start.
    """

    # Compute all pairwise similarities in our unlabeled data.
    pairwise_similarities = squareform(pdist(X, similarity_fn))

    # Obtain the row-wise average for each of our records in X.
    average_similarity = np.mean(pairwise_similarities, axis=0)

    # Isolate and return our best instance for labeling as the
    # record with the greatest average similarity.
    best_coldstart_instance_index = np.argmax(average_similarity)
    return X[best_coldstart_instance_index]


def select_instance(
        X_training: np.ndarray,
        X_uncertainty: np.ndarray,
        similarity_fn: Callable = euclidean_similarity
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core iteration strategy for selecting another record from our unlabeled records.

    Given a set of labeled records (X_training) and unlabeled records with uncertainty
    scores (X_uncertainty), we'd like to identify the best instance in X_uncertainty
    that best balances uncertainty and dissimilarity.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    TODO:
        - Add notebook for Active Learning bake-off (passive vs interactive vs batch vs ranked batch)

    :param X_training:
        Mix of both labeled and unlabeled records.
    :type X_training:
        numpy.ndarray of shape (D + batch_iteration, n_features)

    :param X_uncertainty:
        Unlabeled records to be selected for labeling.
    :type X_uncertainty:
        numpy.ndarray of shape (U - batch_iteration, n_features)

    :param similarity_fn:
        A function that takes two N-length vectors and returns a similarity in range [0, 1].
        Note: any distance function can be a similarity function as 1 / (1 + d)
        where d is the distance.

    :returns:
      - **best_instance_index** *int*
        -- Index of the best index from X chosen to be labelled.
      - **unlabeled_records[best_instance_index]** *(numpy.ndarray of shape (n_features, ))*
        -- a single record from our unlabeled set that is considered the most optimal
            incremental record for including in our query set.
    """

    def max_vector_matrix_distance(arr: np.ndarray, pool: np.ndarray) -> np.float:
        """
        Compute the maximum of pairwise similarity between a flat array and a matrix.

        :param arr:
            Single vector of shape (n_features, ).
        :type arr:
            numpy.ndarray of shape (n_features, )

        :param pool:
            Matrix of shape (n_features, n_records).
        :type pool:
            numpy.ndarray of shape (n_features, n_records)

        :returns:
          - **max_dist** *(np.float)* -- Numeric corresponding to the maximum similarity
            between our vector and matrix.
        """
        return np.max([similarity_fn(arr, x_i) for x_i in pool])

    # Extract the number of labeled and unlabeled records.
    # Note: for unlabeled records, we filter out NaN rows from X_uncertainty
    # because we set them to NaN when, in a previous call to select_instance,
    # we selected that row for labeling.
    n_labeled_records, _ = X_training.shape

    X_uncertainty = X_uncertainty[~np.isnan(X_uncertainty).all(axis=1)]
    n_unlabeled, _ = X_uncertainty.shape

    # Determine our alpha parameter as |U| / (|U| + |D|). Note that because we
    # append to X_training and remove from X_uncertainty within `ranked_batch`,
    # :alpha: is not fixed throughout our model's lifetime.
    alpha = n_unlabeled / (n_unlabeled + n_labeled_records)

    # Isolate our original unlabeled records from their predicted class uncertainty.
    unlabeled_records, uncertainty_scores = X_uncertainty[:, :-1], X_uncertainty[:, -1]

    # Compute pairwise similarity scores from every unlabeled record in unlabeled_records
    # to every record in X_training. The result is an array of shape (n_samples, ).
    similarity_scores = np.apply_along_axis(
        func1d=max_vector_matrix_distance,
        arr=unlabeled_records.T,
        pool=X_training,
        axis=0
    )

    # Compute our final scores, which are a balance between how dissimilar a given record
    # is with the records in X_uncertainty and how uncertain we are about its class.
    scores = alpha * (1 - similarity_scores) + (1 - alpha) * uncertainty_scores

    # Isolate and return our best instance for labeling as the one with the largest score.
    best_instance_index = np.argmax(scores)
    return best_instance_index, unlabeled_records[best_instance_index]


def ranked_batch(classifier: Union[BaseLearner, BaseCommittee],
                 unlabeled: np.ndarray,
                 uncertainty_scores: np.ndarray,
                 n_instances: int) -> np.ndarray:
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

    :returns:
      - **instance_index_ranking** *(numpy.ndarray of shape (n_instances, ))* -- The indices of
        the top n_instances ranked unlabelled samples.

    """

    # Make a local copy of our classifier's training data.
    if classifier.X_training is None:
        labeled = select_cold_start_instance(unlabeled).reshape(1, -1)
    elif classifier.X_training.shape[0] > 0:
        labeled = np.copy(classifier.X_training)

    # Add uncertainty scores to our unlabeled data, and keep a copy of our unlabeled data.
    expanded_uncertainty_scores = np.expand_dims(uncertainty_scores, axis=1)
    unlabeled_uncertainty = np.concatenate((unlabeled, expanded_uncertainty_scores), axis=1)

    # Define our null row, which will be filtered during the select_instance call.
    null_row = np.ones(shape=(unlabeled_uncertainty.shape[1],)) * np.nan

    # Define our record container and the maximum number of records to sample.
    instance_index_ranking = []
    ceiling = np.minimum(unlabeled.shape[0], n_instances)

    for _ in range(ceiling):

        # Receive the instance and corresponding index from our unlabeled copy that scores highest.
        instance_index, instance = select_instance(
            X_training=labeled, X_uncertainty=unlabeled_uncertainty
        )

        # Prepare our most informative instance for concatenation.
        expanded_instance = np.expand_dims(instance, axis=0)

        # Add our instance we've considered for labeling to our labeled set. Although we don't
        # know it's label, we want further iterations to consider the newly-added instance so
        # that we don't query the same instance redundantly.
        labeled = np.concatenate((labeled, expanded_instance), axis=0)

        # We "remove" our instance from the unlabeled set by setting that row to an array
        # of np.nan and filtering within select_instance.
        unlabeled_uncertainty[instance_index] = null_row

        # Finally, append our instance's index to the bottom of our ranking.
        instance_index_ranking.append(instance_index)

    # Return numpy array, not a list.
    return np.array(instance_index_ranking)


def uncertainty_batch_sampling(classifier: Union[BaseLearner, BaseCommittee],
                               X: np.ndarray,
                               n_instances: int = 20,
                               **uncertainty_measure_kwargs: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
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
        numpy.ndarray of shape (n_samples, n_features)

    :param n_instances:
        Number of records to return for labeling from `X`.
    :type n_instances:
        int

    :param uncertainty_measure_kwargs:
        Keyword arguments to be passed for the predict_proba method of the classifier.
    :type uncertainty_measure_kwargs:
        keyword arguments

    :returns:
      - **query_indices** *(numpy.ndarray of shape (n_instances, ))* -- Indices of the
        instances from X chosen to be labelled.
      - **X[query_indices]** *(numpy.ndarray of shape (n_instances, n_features))*
        -- Records from X chosen to be labelled.
    """

    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    query_indices = ranked_batch(classifier, unlabeled=X, uncertainty_scores=uncertainty, n_instances=n_instances)
    return query_indices, X[query_indices]
