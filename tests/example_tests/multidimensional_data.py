import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling, entropy_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.expected_error import expected_error_reduction


class MockClassifier:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.random.randint(0, self.n_classes, shape=(len(X), 1))

    def predict_proba(self, X):
        return np.ones(shape=(len(X), self.n_classes))/self.n_classes


if __name__ == '__main__':
    X_train = np.random.rand(10, 5, 5)
    y_train = np.random.rand(10, 1)
    X_pool = np.random.rand(10, 5, 5)
    y_pool = np.random.rand(10, 1)

    strategies = [margin_sampling, entropy_sampling, uncertainty_batch_sampling]

    for query_strategy in strategies:
        print("testing %s..." % query_strategy.__name__)
        # max margin sampling
        learner = ActiveLearner(
            estimator=MockClassifier(), query_strategy=query_strategy,
            X_training=X_train, y_training=y_train
        )
        learner.query(X_pool)
        learner.teach(X_pool, y_pool)
