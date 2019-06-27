import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI


# generating the data
x1, x2 = np.linspace(0, 10, 11).reshape(-1, 1), np.linspace(0, 10, 11).reshape(-1, 1)
x1, x2 = np.meshgrid(x1, x2)
X = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)

y = np.sin(np.linalg.norm(X, axis=1))/2 - ((10 - np.linalg.norm(X, axis=1))**2)/50 + 2

# assembling initial training set
X_initial, y_initial = X[:10], y[:10]

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)

optimizer = BayesianOptimizer(estimator=GaussianProcessRegressor(kernel=kernel),
                              X_training=X_initial, y_training=y_initial,
                              query_strategy=max_EI)

query_idx, query_inst = optimizer.query(X)
optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx])
