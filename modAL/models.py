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
    predictor: scikit-learn estimator object
        The estimator to be used in the active learning loop.

    uncertainty_measure: function object
        Function providing the uncertainty measure, for instance
        modAL.uncertainty.classifier_uncertainty.

    query_strategy: function object
        Function providing the query strategy for the active learning
        loop, for instance modAL.query.max_uncertainty.

    training_samples: None or numpy.ndarray of shape (n_samples, n_features)
        Initial training samples, if available.

    training_labels: None or numpy.ndarray of shape (n_samples, )
        Initial training labels corresponding to initial training samples

    fit_kwargs: keyword arguments for the fit method

    Attributes
    ----------

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
    ...     training_samples=X_training, training_labels=y_training
    ... )
    >>>
    >>> # the active learning loop
    >>> n_queries = 20
    >>> for idx in range(n_queries):
    ...     query_idx, query_sample = learner.query(iris['data'])
    ...     learner.teach(
    ...         new_sample=iris['data'][query_idx].reshape(1, -1),
    ...         new_label=iris['target'][query_idx].reshape(1, )
    ...     )

    """
    def __init__(
            self,
            predictor,                                           # scikit-learner estimator object
            uncertainty_measure=classifier_uncertainty,          # callable to measure uncertainty
            query_strategy=max_uncertainty, 		             # callable to query labels
            training_samples=None, training_labels=None,	     # initial data if available
            **fit_kwargs                                         # keyword arguments for fitting the initial data
    ):
        assert callable(uncertainty_measure), 'utility_function must be callable'
        assert callable(query_strategy), 'query_function must be callable'

        self.predictor = predictor
        self.uncertainty_measure = uncertainty_measure
        self.query_strategy = query_strategy

        if type(training_samples) == type(None) and type(training_labels) == type(None):
            self.training_data = None
            self.training_labels = None
        elif type(training_samples) != type(None) and type(training_labels) != type(None):
            self.training_data = check_array(training_samples)
            self.training_labels = check_array(training_labels, ensure_2d=False)
            self.fit_to_known(**fit_kwargs)

    def teach(self, new_sample, new_label, **fit_kwargs):
        """
        This function adds the given data to the training examples
        and retrains the predictor with the augmented dataset
        :param new_sample: new training data
        :param new_label: new training labels for the data
        :param fit_kwargs: keyword arguments to be passed to the fit method of classifier
        """
        self.add_training_data(new_sample, new_label)
        self.fit_to_known(**fit_kwargs)

    def add_training_data(self, new_sample, new_label):
        """
        Adds the new data and label to the known data, but does
        not retrain the model.
        :param new_sample:
        :param new_label:
        :return:
        """
        # TODO: get rid of the if clause
        # TODO: test if this works with multiple shapes and types of data

        new_sample, new_label = check_array(new_sample), check_array(new_label, ensure_2d=False)
        assert len(new_sample) == len(new_label), 'the number of new data points and number of labels must match'

        if type(self.training_data) != type(None):
            try:
                self.training_data = np.vstack((self.training_data, new_sample))
                self.training_labels = np.concatenate((self.training_labels, new_label))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')

        else:
            self.training_data = new_sample
            self.training_labels = new_label

    def calculate_uncertainty(self, samples, **uncertainty_measure_kwargs):
        """
        This method calls the utility function provided for ActiveLearner
        on the data passed to it. It is used to measure utilities for each
        data point.
        :param samples: numpy.ndarray, data points for which the utilities should be measured
        :return: utility values for each datapoint as given by the utility function provided
                 for the learner
        """
        check_array(samples)

        return self.uncertainty_measure(self.predictor, samples, **uncertainty_measure_kwargs)

    def fit_to_known(self, **fit_kwargs):
        """
        This method fits self.predictor to the training data and labels
        provided to it so far.
        :param fit_kwargs: keyword arguments to be passed to the fit method of classifier
        """

        self.predictor.fit(self.training_data, self.training_labels, **fit_kwargs)

    def predict(self, samples, **predict_kwargs):
        """
        Interface for the predictor
        :param samples: np.ndarray instances for prediction
        :return: output of the sklearn.base.ClassifierMixin.predict method
        """
        return self.predictor.predict(samples, **predict_kwargs)

    def predict_proba(self, samples, **predict_proba_kwargs):
        """
        Interface for the predict_proba method
        :param samples: np.ndarray of the instances
        :param predict_proba_kwargs: keyword arguments
        :return: output of the sklearn.base.ClassifierMixin.predict_proba method
        """
        return self.predictor.predict_proba(samples, **predict_proba_kwargs)

    def query(self, pool, n_instances=1, **uncertainty_measure_kwargs):
        """
        Finds the n_instances most informative point in the data provided, then
        returns the instances and its indices
        :param pool: np.ndarray, the pool from which the query is selected
        :param n_instances: int, the number of queries
        :return: tuple(query_idx, data[query_idx]), where query_idx is the index of the instance
                 to be queried
        """

        check_array(pool, ensure_2d=True)

        uncertainties = self.calculate_uncertainty(pool, **uncertainty_measure_kwargs)
        query_idx = self.query_strategy(uncertainties, n_instances)
        return query_idx, pool[query_idx]

    def score(self, X, y, **score_kwargs):
        """
        Interface for the score method
        :param X: np.ndarray of the instances to score
        :param y: np.ndarray of the labels
        :param score_kwargs: keyword arguments
        :return: output of the sklearn.base.ClassifierMixin.score method
        """
        return self.predictor.score(X, y, **score_kwargs)


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

        self.learner_list = learner_list
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
            np.concatenate(tuple(learner.predictor.classes_ for learner in self.learner_list), axis=0),
            axis=0
        )
        self.n_classes_ = len(self.classes_)

    def add_and_retrain(self, new_data, new_label, **fit_kwargs):
        self.add_training_data(new_data, new_label)
        self.fit_to_known(**fit_kwargs)

    def add_training_data(self, new_data, new_label):
        for learner in self.learner_list:
            learner.add_training_data(new_data, new_label)
        self._set_classes()

    def calculate_disagreement(self, data, **disagreement_measure_kwargs):
        return self.disagreement_measure(self, data, **disagreement_measure_kwargs)

    def calculate_uncertainty(self, data, **utility_function_kwargs):
        """
        Calculates the uncertainties for every learner in the Committee and returns it
        in the form of a numpy.ndarray
        :param data: numpy.ndarray, data points for which the utilities should be measures
        :return: numpy.ndarray of utilities
        """

        check_array(data, ensure_2d=True)
        uncertainties = np.zeros(shape=(data.shape[0], len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            learner_utility = learner.calculate_uncertainty(data, **utility_function_kwargs)
            uncertainties[:, learner_idx] = learner_utility

        return uncertainties

    def fit_to_known(self, **fit_kwargs):
        for learner in self.learner_list:
            learner.fit_to_known(**fit_kwargs)

    def predict(self, data, **predict_proba_kwargs):
        """
        Predicts the class of the samples by picking
        the average least uncertain prediction.
        """
        # getting average certainties
        proba = self.predict_proba(data, **predict_proba_kwargs)
        # finding the sample-wise max probability
        max_proba_idx = np.argmax(proba, axis=1)
        # translating label indices to labels
        return self.classes_[max_proba_idx]

    def predict_proba(self, data, **predict_proba_kwargs):
        return np.mean(self.vote_proba(data, **predict_proba_kwargs), axis=1)

    def vote(self, data, **predict_kwargs):
        """
        Predicts the labels for the supplied data
        :param data: numpy.ndarray containing the instances to be predicted
        :param predict_kwargs: keyword arguments to be passed for the learners predict method
        :return: numpy.ndarray of shape (n_samples, 1) containing the predictions of all learners
        """

        check_array(data, ensure_2d=True)
        prediction = np.zeros(shape=(data.shape[0], len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            prediction[:, learner_idx] = learner.predict(data, **predict_kwargs)

        return prediction

    def vote_proba(self, data, **predict_proba_kwargs):
        """
        Predicts the probabilities of the classes for each sample and each learner.

        Parameters
        ----------
        data: numpy.ndarray of shape (n_samples, n_features)
            The samples to be predicted by all learners

        predict_proba_kwargs: dict of keyword arguments

        Returns
        -------
        proba: numpy.ndarray of shape (n_samples, n_learners, n_classes)
            Contains the probabilities of each class for each learner and
            each instance

        """

        check_array(data, ensure_2d=True)

        # get dimensions
        n_samples = data.shape[0]
        n_learners = len(self.learner_list)
        proba = np.zeros(shape=(n_samples, n_learners, self.n_classes_))

        # checking if the learners in the Committee know the same set of class labels
        if check_class_labels(*[learner.predictor for learner in self.learner_list]):
            # known class labels are the same for each learner
            # probability prediction is straightforward

            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :] = learner.predict_proba(data, **predict_proba_kwargs)

        else:
            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :] = check_class_proba(
                    proba=learner.predict_proba(data, **predict_proba_kwargs),
                    known_labels=learner.predictor.classes_,
                    all_labels=self.classes_
                )

        return proba

    def query(self, pool, n_instances=1, **disagreement_measure_kwargs):
        """
        Finds the most informative point in the data provided, then
        returns the instance and its index
        :param pool: numpy.ndarray, the pool from which the query is selected
        :return: tuple(query_idx, data[query_idx]), where query_idx is the index of the instance
                 to be queried
        """
        check_array(pool, ensure_2d=True)

        disagreement = self.calculate_disagreement(pool, **disagreement_measure_kwargs)
        query_idx = self.query_strategy(disagreement, n_instances)
        return query_idx, pool[query_idx]


if __name__ == '__main__':
    import doctest
    doctest.testmod()