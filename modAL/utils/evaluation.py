"""
Functions and classes for evaluating the performance of the learners
and comparing them to each other
"""

from copy import deepcopy


class Evaluator:

    def __init__(
            self, learner, performance_metric,
            X_train, y_train,
            X_test, y_test,
            replace=True
    ):
        # model and testing parameters
        self.learner = learner
        self.performance_metric = performance_metric
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.replace = replace

        # model performance
        self.performance = list()

    def evaluate_learner(self, n_queries, replace=True):
        pool = deepcopy(self.X_train)

    def _measure_performance(self):
        pass
