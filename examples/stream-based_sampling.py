"""
In this example the use of ActiveLearner is demonstrated in a pool-based sampling setting.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty

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

# assembling initial training set
n_initial = 5
initial_idx = np.random.choice(range(len(X_full)), size=n_initial, replace=False)
X_train, y_train = X_full[initial_idx], y_full[initial_idx]

# create an ActiveLearner instance
learner = ActiveLearner(
    predictor=RandomForestClassifier(),
    X_initial=X_train, y_initial=y_train
)

"""
The instances are randomly selected one by one, if an instance's uncertainty
is above a threshold, the label is requested and shown to the learner. The
process is continued until the learner reaches a previously defined accuracy.
"""

print('Initial prediction accuracy: %f' % learner.score(X_full, y_full))

while learner.score(X_full, y_full) < 0.95:
    stream_idx = np.random.choice(range(len(X_full)))
    if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:
        learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
        print('Pixel no. %d queried, new accuracy: %f' % (stream_idx, learner.score(X_full, y_full)))