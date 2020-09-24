"""
Learning the shape of an object using uncertainty based sampling.

In this example, we will demonstrate the use of ActiveLearner with
the scikit-learn implementation of the kNN classifier algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

# creating the image
im_width = 500
im_height = 500
data = np.zeros((im_height, im_width))
data[100:im_width-1 - 100, 100:im_height-1 - 100] = 1

# create the pool from the image
X_full = np.transpose(
    [np.tile(np.asarray(range(data.shape[0])), data.shape[1]),
     np.repeat(np.asarray(range(data.shape[1])), data.shape[0])]
)
# map the intensity values against the grid
y_full = np.asarray([data[P[0], P[1]] for P in X_full])
X_pool = deepcopy(X_full)
y_pool = deepcopy(y_full)

# assembling initial training set
initial_idx = [0, im_height-1, im_height*(im_height-1), -1, im_width//2 + im_height//2*im_height]
X_train, y_train = X_pool[initial_idx], y_pool[initial_idx]

# create an ActiveLearner instance
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_train, y_training=y_train
)
initial_prediction = learner.predict_proba(X_full)[:, 1].reshape(im_height, im_width)

n_queries = 100
uncertainty_sampling_accuracy = list()
for round_idx in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(-1, ))
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
    uncertainty_sampling_accuracy.append(learner.score(X_full, y_full))

final_prediction = learner.predict_proba(X_full)[:, 1].reshape(im_height, im_width)

"""
---------------------------------
 comparison with random sampling
---------------------------------
"""


def random_sampling(classsifier, X):
    return np.random.randint(len(X))


X_pool = deepcopy(X_full)
y_pool = deepcopy(y_full)

# learning with randomly selected queries instead of active learning
random_learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=random_sampling,
    X_training=X_train, y_training=y_train
)

random_sampling_accuracy = list()
for round_idx in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    random_learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(-1, ))
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
    random_sampling_accuracy.append(random_learner.score(X_full, y_full))

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(40, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(data)
    plt.title('The shape to learn')
    plt.subplot(1, 4, 2)
    plt.imshow(initial_prediction)
    plt.title('Initial prediction probabilities')
    plt.subplot(1, 4, 3)
    plt.imshow(final_prediction)
    plt.title('Prediction probabilities after query no. %d' % n_queries)
    plt.subplot(1, 4, 4)
    plt.imshow(random_learner.predict_proba(X_full)[:, 1].reshape(im_height, im_width))
    plt.title('Learning with the same amount of randomly selected points')
    plt.show()

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 10))
    plt.plot(list(range(len(uncertainty_sampling_accuracy))), uncertainty_sampling_accuracy, label="uncertainty sampling")
    plt.plot(list(range(len(random_sampling_accuracy))), random_sampling_accuracy, label="random sampling")
    plt.legend()
    plt.show()
