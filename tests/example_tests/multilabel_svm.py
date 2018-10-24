import numpy as np

from modAL.models import ActiveLearner
from modAL.multilabel import SVM_binary_minimum

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

n_samples = 500
X = np.random.normal(size=(n_samples, 2))
y = np.array([[int(x1 > 0), int(x2 > 0)] for x1, x2 in X])

n_initial = 10
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_initial, y_initial = X[initial_idx], y[initial_idx]
X_pool, y_pool = np.delete(X, initial_idx, axis=0), np.delete(y, initial_idx, axis=0)

learner = ActiveLearner(
    estimator=OneVsRestClassifier(LinearSVC()),
    query_strategy=SVM_binary_minimum,
    X_training=X_initial, y_training=y_initial
)

n_queries = 10
for idx in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, -1))
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)