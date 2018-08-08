from modAL.density import euclidean_similarity
from modAL.uncertainty import classifier_uncertainty


def select_instance(X_training, X, X_uncertainty, similarity_fn=euclidean_similarity):
    raise NotImplemented


def ranked_batch(X_training, X, X_uncertainty):
    raise NotImplemented


def uncertainty_batch_sampling(classifier, X, n_instances=1, **uncertainty_measure_kwargs):
    """
    Not implemented yet, coming soon!
    It'll be something like this:

    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    query_idx = ranked_batch(*args, **kwargs)
    return query_idx, X[query_idx]
    """

    raise NotImplemented
