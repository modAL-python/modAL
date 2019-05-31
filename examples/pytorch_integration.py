"""
This example demonstrates how to use the active learning interface with Pytorch.
The example uses Skorch, a scikit learn wrapper of Pytorch.
For more info, see https://skorch.readthedocs.io/en/stable/
"""

import torch
import numpy as np
from keras.datasets import mnist
from torch import nn
from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner

# build class for the skorch API
class Torch_Model(nn.Module):
    def __init__(self,):
        super(Torch_Model, self).__init__()
        self.convs = nn.Sequential(
                                nn.Conv2d(1,32,3),
                                nn.ReLU(),
                                nn.Conv2d(32,64,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.25)
        )
        self.fcs = nn.Sequential(
                                nn.Linear(12*12*64,128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128,10),
        )

    def forward(self, x):
        out = x
        out = self.convs(out)
        out = out.view(-1,12*12*64)
        out = self.fcs(out)
        return out

# create the classifier
classifier = NeuralNetClassifier(Torch_Model,
                                 # max_epochs=100,
                                 criterion=nn.CrossEntropyLoss,
                                 optimizer=torch.optim.Adam,
                                 train_split=None,
                                 verbose=1,
                                 device="cuda")

"""
Data wrangling
1. Reading data from Keras
2. Assembling initial training data for ActiveLearner
3. Generating the pool
"""

# read training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 1, 28, 28).astype('float32') / 255.
X_test = X_test.reshape(10000, 1, 28, 28).astype('float32') / 255.
y_train = y_train.astype('long')
y_test  = y_test.astype('long')

# assemble initial data
n_initial = 1000
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

"""
Training the ActiveLearner
"""

# initialize ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial, y_training=y_initial,
)

# the active learning loop
n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool, n_instances=100)
    print(query_idx)
    learner.teach(X_pool[query_idx], y_pool[query_idx], only_new=True)
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

# the final accuracy score
print(learner.score(X_test, y_test))
