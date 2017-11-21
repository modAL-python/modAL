"""
Functions for evaluating the performance of the learners
and comparing them to each other
"""


class Evaluator:

    def __init__(
            self, learner,
            X_train, y_train,
            X_test, y_test,
            replace=True
    ):
        # model and testing parameters
        self.learner = learner
        self.replace = replace
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # model performance
        self.performance = list()

    def run_training(self, replace=True):
        pass

    def _measure_performance(self):
        pass
