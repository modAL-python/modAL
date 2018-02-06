import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

# visualizing the dataset
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    plt.imshow(data)
    plt.title('The shapes to learn')
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
        predictor=RandomForestClassifier(),
        X_training=X_pool[initial_idx], y_training=y_pool[initial_idx],
        bootstrap_init=True
    )
    learner_list.append(learner)

# assembling the Committee
committee = Committee(learner_list)

# ensemble active learner from the Committee
ensemble_learner = ActiveLearner(
    predictor=committee
)

query_idx, query_instance = ensemble_learner.query(X_pool)

# ...
# ... obtain label from the Oracle ...
# ...

ensemble_learner.teach(X_pool[query_idx], y_pool[query_idx], bootstrap=True)
