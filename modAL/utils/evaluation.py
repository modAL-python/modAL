"""
Functions and classes for evaluating the performance of the learners
and comparing them to each other
"""

import numpy as np
from copy import deepcopy


class Evaluator:

    def __init__(
            self, learner, performance_metric,
            X_train, y_train,
            X_test, y_test
    ):
        """
        :param learner: active learner to evaluate
        :param performance_metric: performance metric to be used
        :param X_train: np.ndarray, training data
        :param y_train: np.ndarray, training labels
        :param X_test: np.ndarray, test data
        :param y_test: np.ndarray, test label
        """
        # model and testing parameters
        self.learner = learner
        self.performance_metric = performance_metric
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # model performance
        self.performance = list()

    def evaluate_learner(self, n_queries, replace=True):
        """
        Trains the given learner
        :param n_queries: int, number of queries to be made
        :param replace: bool, queried data is eliminated from the pool if True
        """
        # clear the performances
        self.performance = list()

        # assemble the pool
        pool_data = deepcopy(self.X_train)
        pool_labels = deepcopy(self.y_train)

        for idx in range(n_queries):
            # query the label
            query_idx, query_instance = self.learner.query(pool_data)
            print(query_idx, query_instance)

            # retrain model with the new label
            self.learner.teach(
                X=query_instance.reshape(1, -1),
                y=pool_labels[query_idx].reshape(-1, )
            )

            # eliminate queried instance from pool if needed
            if not replace:
                pool_data = np.delete(pool_data, query_idx, axis=0)
                pool_labels = np.delete(pool_labels, query_idx)

            # measure performance
            self.performance.append(self.performance_metric(self.y_test, self.learner.predict(self.X_test)))
