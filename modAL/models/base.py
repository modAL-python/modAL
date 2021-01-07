"""
Base classes for active learning algorithms
------------------------------------------
"""

import abc
import sys
import warnings
from typing import Union, Callable, Optional, Tuple, List, Iterator, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y

import scipy.sparse as sp

from modAL.utils.data import data_vstack, data_hstack, modALinput, retrieve_rows

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class BaseLearner(ABC, BaseEstimator):
    """
    Core abstraction in modAL.

    Args:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, modAL.uncertainty.uncertainty_sampling.
        X_training: Initial training samples, if available.
        y_training: Initial training labels corresponding to initial training samples.
        force_all_finite: When True, forces all values of the data finite.
            When False, accepts np.nan and np.inf values.
        bootstrap_init: If initial training data is available, bootstrapping can be done during the first training.
            Useful when building Committee models with bagging.
        on_transformed: Whether to transform samples with the pipeline defined by the estimator
            when applying the query strategy.
        **fit_kwargs: keyword arguments.

    Attributes:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop.
        X_training: If the model hasn't been fitted yet it is None, otherwise it contains the samples
            which the model has been trained on.
        y_training: The labels corresponding to X_training.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 query_strategy: Callable,
                 X_training: Optional[modALinput] = None,
                 y_training: Optional[modALinput] = None,
                 bootstrap_init: bool = False,
                 on_transformed: bool = False,
                 force_all_finite: bool = True,
                 **fit_kwargs
                 ) -> None:
        assert callable(query_strategy), 'query_strategy must be callable'

        self.estimator = estimator
        self.query_strategy = query_strategy
        self.on_transformed = on_transformed

        self.X_training = X_training
        self.y_training = y_training
        if X_training is not None:
            self._fit_to_known(bootstrap=bootstrap_init, **fit_kwargs)

        assert isinstance(force_all_finite, bool), 'force_all_finite must be a bool'
        self.force_all_finite = force_all_finite

    def _add_training_data(self, X: modALinput, y: modALinput) -> None:
        """
        Adds the new data and label to the known data, but does not retrain the model.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.

        Note:
            If the classifier has been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        check_X_y(X, y, accept_sparse=True, ensure_2d=False, allow_nd=True, multi_output=True, dtype=None,
                  force_all_finite=self.force_all_finite)

        if self.X_training is None:
            self.X_training = X
            self.y_training = y
        else:
            try:
                self.X_training = data_vstack((self.X_training, X))
                self.y_training = data_vstack((self.y_training, y))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')

    def transform_without_estimating(self, X: modALinput) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Transforms the data as supplied to the estimator.

        * In case the estimator is an skearn pipeline, it applies all pipeline components but the last one.
        * In case the estimator is an ensemble, it concatenates the transformations for each classfier
            (pipeline) in the ensemble.
        * Otherwise returns the non-transformed dataset X
        Args:
            X: dataset to be transformed

        Returns:
            Transformed data set
        """
        Xt = []
        pipes = [self.estimator]

        if isinstance(self.estimator, _BaseHeterogeneousEnsemble):
            pipes = self.estimator.estimators_

        ################################
        # transform data with pipelines used by estimator
        for pipe in pipes:
            if isinstance(pipe, Pipeline):
                # NOTE: The used pipeline class might be an extension to sklearn's!
                #       Create a new instance of the used pipeline class with all
                #       components but the final estimator, which is replaced by an empty (passthrough) component.
                #       This prevents any special handling of the final transformation pipe, which is usually
                #       expected to be an estimator.
                transformation_pipe = pipe.__class__(steps=[*pipe.steps[:-1], ('passthrough', 'passthrough')])
                Xt.append(transformation_pipe.transform(X))

        # in case no transformation pipelines are used by the estimator,
        # return the original, non-transfored data
        if not Xt:
            return X

        ################################
        # concatenate all transformations and return
        return data_hstack(Xt)

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        """
        Fits self.estimator to the training data and labels provided to it so far.

        Args:
            bootstrap: If True, the method trains the model on a set bootstrapped from the known training instances.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """
        if not bootstrap:
            self.estimator.fit(self.X_training, self.y_training, **fit_kwargs)
        else:
            n_instances = self.X_training.shape[0]
            bootstrap_idx = np.random.choice(range(n_instances), n_instances, replace=True)
            self.estimator.fit(self.X_training[bootstrap_idx], self.y_training[bootstrap_idx], **fit_kwargs)

        return self

    def _fit_on_new(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        """
        Fits self.estimator to the given data and labels.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, the method trains the model on a set bootstrapped from X.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """
        check_X_y(X, y, accept_sparse=True, ensure_2d=False, allow_nd=True, multi_output=True, dtype=None,
                  force_all_finite=self.force_all_finite)

        if not bootstrap:
            self.estimator.fit(X, y, **fit_kwargs)
        else:
            bootstrap_idx = np.random.choice(range(X.shape[0]), X.shape[0], replace=True)
            self.estimator.fit(X[bootstrap_idx], y[bootstrap_idx])

        return self

    def fit(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied data, then stores it
        internally for the active learning loop.

        Args:
            X: The samples to be fitted.
            y: The corresponding labels.
            bootstrap: If true, trains the estimator on a set bootstrapped from X.
                Useful for building Committee models with bagging.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Note:
            When using scikit-learn estimators, calling this method will make the ActiveLearner forget all training data
            it has seen!

        Returns:
            self
        """
        check_X_y(X, y, accept_sparse=True, ensure_2d=False, allow_nd=True, multi_output=True, dtype=None,
                  force_all_finite=self.force_all_finite)
        self.X_training, self.y_training = X, y
        return self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)

    def predict(self, X: modALinput, **predict_kwargs) -> Any:
        """
        Estimator predictions for X. Interface with the predict method of the estimator.

        Args:
            X: The samples to be predicted.
            **predict_kwargs: Keyword arguments to be passed to the predict method of the estimator.

        Returns:
            Estimator predictions for X.
        """
        return self.estimator.predict(X, **predict_kwargs)

    def predict_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Class probabilities if the predictor is a classifier. Interface with the predict_proba method of the classifier.

        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the predict_proba method of the classifier.

        Returns:
            Class probabilities for X.
        """
        return self.estimator.predict_proba(X, **predict_proba_kwargs)

    def query(self, X_pool, *query_args, **query_kwargs) -> Union[Tuple, modALinput]:
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.

        Args:
            X_pool: Pool of unlabeled instances to retrieve most informative instances from
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~modAL.uncertainty.uncertainty_sampling`, it is the pool of samples from which the query strategy
                should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.

        Returns:
            Value of the query_strategy function. Should be the indices of the instances from the pool chosen to be
            labelled and the instances themselves. Can be different in other cases, for instance only the instance to be
            labelled upon query synthesis.
        """
        query_result = self.query_strategy(self, X_pool, *query_args, **query_kwargs)

        if isinstance(query_result, tuple):
            warnings.warn("Query strategies should no longer return the selected instances, "
                          "this is now handled by the query method. "
                          "Please return only the indices of the selected instances.", DeprecationWarning)
            return query_result

        return query_result, retrieve_rows(X_pool, query_result)

    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        """
        Interface for the score method of the predictor.

        Args:
            X: The samples for which prediction accuracy is to be calculated.
            y: Ground truth labels for X.
            **score_kwargs: Keyword arguments to be passed to the .score() method of the predictor.

        Returns:
            The score of the predictor.
        """
        return self.estimator.score(X, y, **score_kwargs)

    @abc.abstractmethod
    def teach(self, *args, **kwargs) -> None:
        pass


class BaseCommittee(ABC, BaseEstimator):
    """
    Base class for query-by-committee setup.

    Args:
        learner_list: List of ActiveLearner objects to form committee.
        query_strategy: Function to query labels.
        on_transformed: Whether to transform samples with the pipeline defined by each learner's estimator
            when applying the query strategy.
    """
    def __init__(self, learner_list: List[BaseLearner], query_strategy: Callable, on_transformed: bool = False) -> None:
        assert type(learner_list) == list, 'learners must be supplied in a list'

        self.learner_list = learner_list
        self.query_strategy = query_strategy
        self.on_transformed = on_transformed
        # TODO: update training data when using fit() and teach() methods
        self.X_training = None

    def __iter__(self) -> Iterator[BaseLearner]:
        for learner in self.learner_list:
            yield learner

    def __len__(self) -> int:
        return len(self.learner_list)

    def _add_training_data(self, X: modALinput, y: modALinput) -> None:
        """
        Adds the new data and label to the known data for each learner, but does not retrain the model.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.

        Note:
            If the learners have been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        for learner in self.learner_list:
            learner._add_training_data(X, y)

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> None:
        """
        Fits all learners to the training data and labels provided to it so far.

        Args:
            bootstrap: If True, each estimator is trained on a bootstrapped dataset. Useful when
                using bagging to build the ensemble.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        for learner in self.learner_list:
            learner._fit_to_known(bootstrap=bootstrap, **fit_kwargs)

    def _fit_on_new(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> None:
        """
        Fits all learners to the given data and labels.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, the method trains the model on a set bootstrapped from X.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        for learner in self.learner_list:
            learner._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)

    def fit(self, X: modALinput, y: modALinput, **fit_kwargs) -> 'BaseCommittee':
        """
        Fits every learner to a subset sampled with replacement from X. Calling this method makes the learner forget the
        data it has seen up until this point and replaces it with X! If you would like to perform bootstrapping on each
        learner using the data it has seen, use the method .rebag()!

        Calling this method makes the learner forget the data it has seen up until this point and replaces it with X!

        Args:
            X: The samples to be fitted on.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        for learner in self.learner_list:
            learner.fit(X, y, **fit_kwargs)

        return self

    def transform_without_estimating(self, X: modALinput) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Transforms the data as supplied to each learner's estimator and concatenates transformations.
        Args:
            X: dataset to be transformed

        Returns:
            Transformed data set
        """
        return data_hstack([learner.transform_without_estimating(X) for learner in self.learner_list])

    def query(self, X_pool, *query_args, **query_kwargs) -> Union[Tuple, modALinput]:
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.

        Args:
            X_pool: Pool of unlabeled instances to retrieve most informative instances from
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~modAL.disagreement.max_disagreement_sampling`, it is the pool of samples from which the query.
                strategy should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.

        Returns:
            Return value of the query_strategy function. Should be the indices of the instances from the pool chosen to
            be labelled and the instances themselves. Can be different in other cases, for instance only the instance to
            be labelled upon query synthesis.
        """
        query_result = self.query_strategy(self, X_pool, *query_args, **query_kwargs)

        if isinstance(query_result, tuple):
            warnings.warn("Query strategies should no longer return the selected instances, "
                          "this is now handled by the query method. "
                          "Please return only the indices of the selected instances", DeprecationWarning)
            return query_result

        return query_result, retrieve_rows(X_pool, query_result)

    def rebag(self, **fit_kwargs) -> None:
        """
        Refits every learner with a dataset bootstrapped from its training instances. Contrary to .bag(), it bootstraps
        the training data for each learner based on its own examples.

        Todo:
            Where is .bag()?

        Args:
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._fit_to_known(bootstrap=True, **fit_kwargs)

    def teach(self, X: modALinput, y: modALinput, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data for each learner and retrains learners with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._add_training_data(X, y)
        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)

    @abc.abstractmethod
    def predict(self, X: modALinput) -> Any:
        pass

    @abc.abstractmethod
    def vote(self, X: modALinput) -> Any:  # TODO: clarify typing
        pass