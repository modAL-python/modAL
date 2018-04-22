import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import PI, EI, UCB, max_PI, max_EI, max_UCB


# generating the data
X = np.linspace(0, 20, 1000).reshape(-1, 1)
y = np.sin(X)/2 - ((10 - X)**2)/50 + 2

# assembling initial training set
X_initial, y_initial = X[150].reshape(1, -1), y[150].reshape(1, -1)

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)

# initializing the optimizer
optimizer = BayesianOptimizer(
    estimator=GaussianProcessRegressor(kernel=kernel),
    X_training=X_initial, y_training=y_initial,
    query_strategy=max_EI
)

# plotting the initial estimation
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(30, 6))
    for n_query in range(5):
        # plot current prediction
        plt.subplot(2, 5, n_query + 1)
        plt.title('Query no. %d' %(n_query + 1))
        if n_query == 0:
            plt.ylabel('Predictions')
        plt.ylim([-1.5, 3])
        pred, std = optimizer.predict(X.reshape(-1, 1), return_std=True)
        utility = EI(optimizer, X)
        plt.plot(X, pred)
        plt.fill_between(X.reshape(-1, ), pred.reshape(-1, ) - std, pred.reshape(-1, ) + std, alpha=0.2)
        plt.plot(X, y, c='k', linewidth=3)
        # plotting acquired values
        plt.scatter(optimizer.X_training[-1], optimizer.y_training[-1], c='w', s=40, zorder=20)
        plt.scatter(optimizer.X_training, optimizer.y_training, c='k', s=80, zorder=1)

        plt.subplot(2, 5, 5 + n_query + 1)
        if n_query == 0:
            plt.ylabel('Expected improvement')
        plt.plot(X, 5*utility, c='r')
        plt.ylim([-0.1, 1])

        # query
        query_idx, query_inst = optimizer.query(X)
        optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
    plt.show()
