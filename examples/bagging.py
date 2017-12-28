"""
This example shows how to build models with bagging using the Committee model.
"""

import numpy as np
from scipy import misc
from itertools import product
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner, Committee

# creating the dataset
im_width = 500
im_height = 500
data = np.zeros((im_height, im_width))
# each disk is coded as a triple (x, y, r), where x and y are the centers and r is the radius
disks = [(150, 150, 80), (200, 380, 50), (360, 200, 100)]
for i, j in product(range(im_width), range(im_height)):
    for x, y, r in disks:
        if (x-i)**2 + (y-j)**2 < r**2:
            data[i, j] = 1

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 10))
    plt.imshow(data)
    plt.title('The shapes to predict')
    plt.show()

# create the pool from the image
X_pool = np.transpose(
    [np.tile(np.asarray(range(data.shape[0])), data.shape[1]),
     np.repeat(np.asarray(range(data.shape[1])), data.shape[0])]
)
# map the intensity values against the grid
y_pool = np.asarray([data[P[0], P[1]] for P in X_pool])

# initial training data: 100 random pixels
initial_idx = np.random.choice(range(len(X_pool)), size=100)

# initializing the learners
n_learners = 3
learner_list = []
for _ in range(n_learners):
    learner = ActiveLearner(
        predictor=KNeighborsClassifier(),
        X_initial=X_pool[initial_idx], y_initial=y_pool[initial_idx],
        bootstrap_init=True
    )
    learner_list.append(learner)

# assembling the Committee
committee = Committee(learner_list)

# visualizing every learner in the Committee
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10*n_learners, 10))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_learners, learner_idx+1)
        plt.imshow(learner.predict(X_pool).reshape(im_height, im_width))
        plt.title('Learner no. %d' % learner_idx)
    plt.show()

# rebagging the data
committee.rebag()

# visualizing the learners in the retrained Committee
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10*n_learners, 10))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_learners, learner_idx+1)
        plt.imshow(learner.predict(X_pool).reshape(im_height, im_width))
        plt.title('Learner no. %d' % learner_idx)
    plt.show()