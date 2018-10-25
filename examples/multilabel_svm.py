import numpy as np
import matplotlib.pyplot as plt

from modAL.models import ActiveLearner
from modAL.multilabel import SVM_binary_minimum

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

n_samples = 500
X = np.random.normal(size=(n_samples, 2))
y = np.array([[int(x1 > 0), int(x2 > 0)] for x1, x2 in X])

n_initial = 10
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_initial, y_initial = X[initial_idx], y[initial_idx]
X_pool, y_pool = np.delete(X, initial_idx, axis=0), np.delete(y, initial_idx, axis=0)

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c='k', s=20)
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1],
                facecolors='none', edgecolors='b', s=50, linewidths=2, label='class 1')
    plt.scatter(X[y[:, 1] == 1, 0], X[y[:, 1] == 1, 1],
                facecolors='none', edgecolors='r', s=100, linewidths=2, label='class 2')
    plt.legend()
    #plt.show()

learner = ActiveLearner(
    estimator=OneVsRestClassifier(SVC(probability=True)),
    query_strategy=SVM_binary_minimum,
    X_training=X_initial, y_training=y_initial
)

learner.query(X_pool)