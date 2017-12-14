"""
Core models for active learning algorithms.
"""

import numpy as np
from sklearn.utils import check_array
from modAL.utils.validation import check_class_labels, check_class_proba
from modAL.uncertainty import classifier_uncertainty
from modAL.disagreement import vote_entropy
from modAL.query import max_uncertainty


class ActiveLearner:
    """
    This class is an abstract model of a general active learning algorithm.

    Parameters
    ----------
    predictor: scikit-learn estimator
        The estimator to be used in the active learning loop.

    uncertainty_measure: function
        Function providing the uncertainty measure, for instance
        modAL.uncertainty.classifier_uncertainty.

    query_strategy: function
        Function providing the query strategy for the active learning
        loop, for instance modAL.query.max_uncertainty.

    training_samples: None or numpy.ndarray of shape (n_samples, n_features)
        Initial training samples, if available.

    training_labels: None or numpy.ndarray of shape (n_samples, )
        Initial training labels corresponding to initial training samples

    fit_kwargs: keyword arguments for the fit method

    Attributes
    ----------
    predictor: scikit-learn estimator
        The estimator to be used in the active learning loop.

    uncertainty_measure: function
        Function providing the uncertainty measure, for instance
        modAL.uncertainty.classifier_uncertainty.

    query_strategy: function
        Function providing the query strategy for the active learning
        loop, for instance modAL.query.max_uncertainty.

    _training_samples: None numpy.ndarray of shape (n_samples, n_features)
        If the model hasn't been fitted yet: None
        If the model has been fitted already: numpy.ndarray containing the
        samples which the model has been trained on

    _training_labels: None or numpy.ndarray of shape (n_samples, )
        If the model hasn't been fitted yet: None
        If the model has been fitted already: numpy.ndarray containing the
        labels corresponding to _training_samples

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from modAL.models import ActiveLearner
    >>>
    >>> iris = load_iris()
    >>> # give initial training examples
    >>> X_training = iris['data'][[0, 50, 100]]
    >>> y_training = iris['target'][[0, 50, 100]]
    >>>
    >>> # initialize active learner
    >>> learner = ActiveLearner(
    ...     predictor=RandomForestClassifier(),
    ...     X_initial=X_training, y_initial=y_training
    ... )
    >>>
    >>> # the active learning loop
    >>> n_queries = 20
    >>> for idx in range(n_queries):
    ...     query_idx, query_sample = learner.query(iris['data'])
    ...     learner.teach(
    ...         X=iris['data'][query_idx].reshape(1, -1),
    ...         y=iris['target'][query_idx].reshape(1, )
    ...     )

    """
    def __init__(
            self,
            predictor,                                           # scikit-learner estimator object
            uncertainty_measure=classifier_uncertainty,          # callable to measure uncertainty
            query_strategy=max_uncertainty, 		             # callable to query labels
            X_initial=None, y_initial=None,	                     # initial data if available
            **fit_kwargs                                         # keyword arguments for fitting the initial data
    ):
        assert callable(uncertainty_measure), 'utility_function must be callable'
        assert callable(query_strategy), 'query_function must be callable'

        self._predictor = predictor
        self.uncertainty_measure = uncertainty_measure
        self.query_strategy = query_strategy

        if type(X_initial) == type(None) and type(y_initial) == type(None):
            self._X_training = None
            self._y_training = None
        elif type(X_initial) != type(None) and type(y_initial) != type(None):
            self._X_training = check_array(X_initial)
            self._y_training = check_array(y_initial, ensure_2d=False)
            self.fit_to_known(**fit_kwargs)

    def add_training_data(self, X, y):
        """
        Adds the new data and label to the known data, but does
        not retrain the model. Used internally of in stream based
        active learning scenarios.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The new samples for which the labels are supplied
            by the expert.

        y: numpy.ndarray of shape (n_samples, )
            Labels corresponding to the new instances in X.

        Note
        ----
        If the classifier has been fitted, the features in X
        have to agree with the training samples which the
        classifier has seen.
        """

        X, y = check_array(X), check_array(y, ensure_2d=False)
        assert len(X) == len(y), 'the number of new data points and number of labels must match'

        if type(self._X_training) != type(None):
            try:
                self._X_training = np.vstack((self._X_training, X))
                self._y_training = np.concatenate((self._y_training, y))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')

        else:
            self._X_training = X
            self._y_training = y

    def calculate_uncertainty(self, X, **uncertainty_measure_kwargs):
        """
        This method calls the uncertainty measure function provided
        for ActiveLearner upon initialization on the samples passed
        to it. It is used to measure uncertainty for each data point.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The samples for which the uncertainty of prediction is
            to be calculated.

        uncertainty_measure_kwargs: keyword arguments
            Keyword arguments to be passed to the uncertainty measure
            function.

        Returns
        -------
        uncertainty: numpy.ndarray of shape (n_samples, )
            Contains the uncertainties for the predictions on sample X
        """
        check_array(X)
        return self.uncertainty_measure(self._predictor, X, **uncertainty_measure_kwargs)

    def fit_to_known(self, **fit_kwargs):
        """
        This method fits self.predictor to the training data and labels
        provided to it so far. Used internally or in stream based active
        learning scenarios.

        Parameters
        ----------
        fit_kwargs: keyword arguments
            Keyword arguments to be passed to the fit method of the predictor.

        """
        self._predictor.fit(self._X_training, self._y_training, **fit_kwargs)

    def fit(self, X, y, **fit_kwargs):
        """
        Interface for the fit method of the predictor. Fits the predictor
        to the supplied data, then stores it internally for the active
        learning loop.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The samples to be fitted.

        y: numpy.ndarray of shape (n_samples, )
            The corresponding labels.

        fit_kwargs: keyword arguments
            Keyword arguments to be passed to the fit method of the predictor.
        """
        self._predictor.fit(X, y, **fit_kwargs)
        self._X_training = X
        self._y_training = y

    def predict(self, X, **predict_kwargs):
        """
        Estimator predictions for X. Interface with the predict
        method of the estimator.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        pred: numpy.ndarray of shape (n_samples, )
            Estimator predictions for X.
        """
        return self._predictor.predict(X, **predict_kwargs)

    def predict_proba(self, X, **predict_proba_kwargs):
        """
        Class probabilities if the predictor is a classifier.
        Interface with the predict_proba method of the classifier.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The samples for which the class probabilities are
            to be predicted.

        Returns
        -------
        proba: numpy.ndarray of shape (n_samples, n_classes)
            Class probabilities for X.
        """
        return self._predictor.predict_proba(X, **predict_proba_kwargs)

    def query(self, X_pool, n_instances=1, **uncertainty_measure_kwargs):
        """
        Finds the n_instances most informative point in the data provided, then
        returns the instances and its indices
        :param X_pool: np.ndarray, the pool from which the query is selected
        :param n_instances: int, the number of queries
        :return: tuple(query_idx, data[query_idx]), where query_idx is the index of the instance
                 to be queried
        """

        check_array(X_pool, ensure_2d=True)

        uncertainties = self.calculate_uncertainty(X_pool, **uncertainty_measure_kwargs)
        query_idx = self.query_strategy(uncertainties, n_instances)
        return query_idx, X_pool[query_idx]

    def score(self, X, y, **score_kwargs):
        """
        Interface for the score method
        :param X: np.ndarray of the instances to score
        :param y: np.ndarray of the labels
        :param score_kwargs: keyword arguments
        :return: output of the sklearn.base.ClassifierMixin.score method
        """
        return self._predictor.score(X, y, **score_kwargs)

    def teach(self, X, y, **fit_kwargs):
        """
        This function adds X and y to the known training data
        and retrains the predictor with the augmented dataset.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The new samples for which the labels are supplied
            by the expert.

        y: numpy.ndarray of shape (n_samples, )
            Labels corresponding to the new instances in X.

        fit_kwargs: keyword arguments
            Keyword arguments to be passed to the fit method
            of the predictor.
        """
        self.add_training_data(X, y)
        self.fit_to_known(**fit_kwargs)


class Committee:
    """
    This class is an abstract model of a committee-based active learning algorithm.
    """
    def __init__(
            self,
            learner_list,                                        # list of ActiveLearner objects
            disagreement_measure=vote_entropy,                   # callable to measure disagreement
            query_strategy=max_uncertainty                       # callable to query labels

    ):
        """
        :param learner_list: list of ActiveLearners
        """
        assert type(learner_list) == list, 'learners must be supplied in a list'

        self._learner_list = learner_list
        self.disagreement_measure = disagreement_measure
        self.query_strategy = query_strategy

        self._set_classes()

    def _set_classes(self):
        """
        Checks the known class labels by each learner,
        merges the labels and returns a mapping which
        maps the learner's classes to the complete label
        list
        """

        # assemble the list of known classes from each learner
        self.classes_ = np.unique(
            np.concatenate(tuple(learner._predictor.classes_ for learner in self._learner_list), axis=0),
            axis=0
        )
        self.n_classes_ = len(self.classes_)

    def add_training_data(self, X, y):
        for learner in self._learner_list:
            learner.add_training_data(X, y)
        self._set_classes()

    def calculate_disagreement(self, X, **disagreement_measure_kwargs):
        return self.disagreement_measure(self, X, **disagreement_measure_kwargs)

    def calculate_uncertainty(self, X, **utility_function_kwargs):
        """
        Calculates the uncertainties for every learner in the Committee and returns it
        in the form of a numpy.ndarray
        :param X: numpy.ndarray, data points for which the utilities should be measures
        :return: numpy.ndarray of utilities
        """

        check_array(X, ensure_2d=True)
        uncertainties = np.zeros(shape=(X.shape[0], len(self._learner_list)))

        for learner_idx, learner in enumerate(self._learner_list):
            learner_utility = learner.calculate_uncertainty(X, **utility_function_kwargs)
            uncertainties[:, learner_idx] = learner_utility

        return uncertainties

    def fit_to_known(self, **fit_kwargs):
        for learner in self._learner_list:
            learner.fit_to_known(**fit_kwargs)

    def predict(self, X, **predict_proba_kwargs):
        """
        Predicts the class of the samples by picking
        the average least uncertain prediction.
        """
        # getting average certainties
        proba = self.predict_proba(X, **predict_proba_kwargs)
        # finding the sample-wise max probability
        max_proba_idx = np.argmax(proba, axis=1)
        # translating label indices to labels
        return self.classes_[max_proba_idx]

    def predict_proba(self, X, **predict_proba_kwargs):
        return np.mean(self.vote_proba(X, **predict_proba_kwargs), axis=1)

    def vote(self, X, **predict_kwargs):
        """
        Predicts the labels for the supplied data
        :param X: numpy.ndarray containing the instances to be predicted
        :param predict_kwargs: keyword arguments to be passed for the learners predict method
        :return: numpy.ndarray of shape (n_samples, 1) containing the predictions of all learners
        """

        check_array(X, ensure_2d=True)
        prediction = np.zeros(shape=(X.shape[0], len(self._learner_list)))

        for learner_idx, learner in enumerate(self._learner_list):
            prediction[:, learner_idx] = learner.predict(X, **predict_kwargs)

        return prediction

    def vote_proba(self, X, **predict_proba_kwargs):
        """
        Predicts the probabilities of the classes for each sample and each learner.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The samples to be predicted by all learners

        predict_proba_kwargs: dict of keyword arguments

        Returns
        -------
        proba: numpy.ndarray of shape (n_samples, n_learners, n_classes)
            Contains the probabilities of each class for each learner and
            each instance

        """

        check_array(X, ensure_2d=True)

        # get dimensions
        n_samples = X.shape[0]
        n_learners = len(self._learner_list)
        proba = np.zeros(shape=(n_samples, n_learners, self.n_classes_))

        # checking if the learners in the Committee know the same set of class labels
        if check_class_labels(*[learner.predictor for learner in self._learner_list]):
            # known class labels are the same for each learner
            # probability prediction is straightforward

            for learner_idx, learner in enumerate(self._learner_list):
                proba[:, learner_idx, :] = learner.predict_proba(X, **predict_proba_kwargs)

        else:
            for learner_idx, learner in enumerate(self._learner_list):
                proba[:, learner_idx, :] = check_class_proba(
                    proba=learner.predict_proba(X, **predict_proba_kwargs),
                    known_labels=learner.predictor.classes_,
                    all_labels=self.classes_
                )

        return proba

    def query(self, X_pool, n_instances=1, **disagreement_measure_kwargs):
        """
        Finds the most informative point in the data provided, then
        returns the instance and its index
        :param X_pool: numpy.ndarray, the pool from which the query is selected
        :return: tuple(query_idx, data[query_idx]), where query_idx is the index of the instance
                 to be queried
        """
        check_array(X_pool, ensure_2d=True)

        disagreement = self.calculate_disagreement(X_pool, **disagreement_measure_kwargs)
        query_idx = self.query_strategy(disagreement, n_instances)
        return query_idx, X_pool[query_idx]

    def teach(self, X, y, **fit_kwargs):
        self.add_training_data(X, y)
        self.fit_to_known(**fit_kwargs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()