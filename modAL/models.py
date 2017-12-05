"""
Core models for active learning algorithms.
"""

import numpy as np
from sklearn.utils import check_array


class ActiveLearner:
    """
    This class is an abstract model of a general active learning algorithm.
    """
    def __init__(
            self,
            predictor, utility_function, 					# building blocks of the learner
            training_data=None, training_labels=None			# initial data if available
    ):
        """
        :param predictor: an instance of the predictor
        :param utility_function: function to calculate utilities
        :param training_data: initial training data if available
        :param training_labels: labels corresponding to the initial training data
        """

        # TODO: add keyword argument handling for the fit_to_known() method

        assert callable(utility_function), 'utility_function must be callable'

        self.predictor = predictor
        self.utility_function = utility_function

        if type(training_data) == type(None) and type(training_labels) == type(None):
            self.training_data = None
            self.training_labels = None
        elif type(training_data) != type(None) and type(training_labels) != type(None):
            self.training_data = check_array(training_data)
            self.training_labels = check_array(training_labels, ensure_2d=False)

        if (type(training_data) != type(None)) and (type(training_labels) != type(None)):
            self.fit_to_known()

    def calculate_utility(self, data):
        """
        This method calls the utility function provided for ActiveLearner
        on the data passed to it. It is used to measure utilities for each
        data point.
        :param data: numpy.ndarray, data points for which the utilities should be measured
        :return: utility values for each datapoint as given by the utility function provided
                 for the learner
        """
        check_array(data)

        return self.utility_function(self.predictor, data)

    def query(self, data, n_instances=1):
        """
        Finds the n_instances most informative point in the data provided, then
        returns the instances and its indices
        :param data: np.ndarray, the pool from which the query is selected
        :param n_instances: int, the number of queries
        :return: tuple(query_idx, data[query_idx]), where query_idx is the index of the instance
                 to be queried
        """

        check_array(data, ensure_2d=True)

        utilities = self.calculate_utility(data)
        query_idx = np.argpartition(-utilities, n_instances)[:n_instances]
        return query_idx, data[query_idx]

    def fit_to_known(self, **fit_kwargs):
        """
        This method fits self.predictor to the training data and labels
        provided to it so far.
        :param fit_kwargs: keyword arguments to be passed to the fit method of classifier
        """
        self.predictor.fit(self.training_data, self.training_labels, **fit_kwargs)

    def add_and_retrain(self, new_data, new_label, **fit_kwargs):
        """
        This function adds the given data to the training examples
        and retrains the predictor with the augmented dataset
        :param new_data: new training data
        :param new_label: new training labels for the data
        :param fit_kwargs: keyword arguments to be passed to the fit method of classifier
        """
        self.add_training_data(new_data, new_label)
        self.fit_to_known(**fit_kwargs)

    def add_training_data(self, new_data, new_label):
        """
        Adds the new data and label to the known data, but does
        not retrain the model.
        :param new_data:
        :param new_label:
        :return:
        """
        # TODO: get rid of the if clause
        # TODO: test if this works with multiple shapes and types of data

        new_data, new_label = check_array(new_data), check_array(new_label, ensure_2d=False)
        assert len(new_data) == len(new_label), 'the number of new data points and number of labels must match'

        if type(self.training_data) != type(None):
            try:
                self.training_data = np.vstack((self.training_data, new_data))
                self.training_labels = np.concatenate((self.training_labels, new_label))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')

        else:
            self.training_data = new_data
            self.training_labels = new_label

    def predict(self, data, **predict_kwargs):
        """
        Interface for the predictor
        :param data: np.ndarray instances for prediction
        :return: output of the sklearn.base.ClassifierMixin.predict method
        """
        return self.predictor.predict(data, **predict_kwargs)

    def predict_proba(self, data, **predict_proba_kwargs):
        """
        Interface for the predict_proba method
        :param data: np.ndarray of the instances
        :param predict_proba_kwargs: keyword arguments
        :return: output of the sklearn.base.ClassifierMixin.predict_proba method
        """
        return self.predictor.predict_proba(data, **predict_proba_kwargs)

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
    def __init__(self, learner_list):
        """
        :param learner_list: list of ActiveLearners
        """
        assert type(learner_list) == list, 'learners must be supplied in a list'

        self.learner_list = learner_list

    def calculate_utility(self, data):
        """
        Calculates the utilities for every learner in the Committee and returns it
        in the form of a numpy.ndarray
        :param data: numpy.ndarray, data points for which the utilities should be measures
        :return: numpy.ndarray of utilities
        """

        check_array(data, ensure_2d=True)
        utilities = np.zeros(shape=(data.shape[0], len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            learner_utility = learner.calculate_utility(data)
            utilities[:, learner_idx] = learner_utility

        return utilities

    def query(self, data):
        """
        Finds the most informative point in the data provided, then
        returns the instance and its index
        :param data: numpy.ndarray, the pool from which the query is selected
        :return: tuple(query_idx, data[query_idx]), where query_idx is the index of the instance
                 to be queried
        """
        pass

    def add_and_retrain(self, new_data, new_label):
        pass

    def add_training_data(self, new_data, new_label):
        pass

    def predict(self, data):
        pass

    def predict_proba(self, data):
        pass
