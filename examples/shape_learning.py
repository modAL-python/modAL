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
    predictor=RandomForestClassifier(),
    X_training=X_train, y_training=y_train
)
initial_prediction = learner.predict_proba(X_full)[:, 1].reshape(im_height, im_width)

n_queries = 100
for round_idx in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(-1, ))
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

final_prediction = learner.predict_proba(X_full)[:, 1].reshape(im_height, im_width)

# learning with randomly selected queries instead of active learning
random_idx = initial_idx + list(np.random.choice(range(len(X_full)), n_queries, replace=False))
X_train, y_train = X_full[initial_idx], y_full[initial_idx]
random_learner = ActiveLearner(
    predictor=RandomForestClassifier(),
    X_training=X_train, y_training=y_train
)

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
