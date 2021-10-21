from functools import partial

import numpy as np
from modAL.acquisition import (max_EI, max_PI, max_UCB, optimizer_EI,
                               optimizer_PI, optimizer_UCB)
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# generating the data
X = np.linspace(0, 20, 1000).reshape(-1, 1)
y = np.sin(X)/2 - ((10 - X)**2)/50 + 2

# assembling initial training set
X_initial, y_initial = X[150].reshape(1, -1), y[150].reshape(1, -1)

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)

tr = 0.1
PI_tr = partial(optimizer_PI, tradeoff=tr)
PI_tr.__name__ = 'PI, tradeoff = %1.1f' % tr
max_PI_tr = partial(max_PI, tradeoff=tr)

acquisitions = zip(
    [PI_tr, optimizer_EI, optimizer_UCB],
    [max_PI_tr, max_EI, max_UCB],
)

for acquisition, query_strategy in acquisitions:

    # initializing the optimizer
    optimizer = BayesianOptimizer(
        estimator=GaussianProcessRegressor(kernel=kernel),
        X_training=X_initial, y_training=y_initial,
        query_strategy=query_strategy
    )

    for n_query in range(5):
        query_idx, query_inst = optimizer.query(X)
        optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
