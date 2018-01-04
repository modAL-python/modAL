import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner, CommitteeRegressor

# generating the data
X = np.concatenate((np.random.rand(100)-1, np.random.rand(100)))
y = np.abs(X)

# initializing the regressors
n_initial = 5
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

initial_idx = list()
initial_idx.append(np.random.choice(range(100), size=n_initial, replace=False))
initial_idx.append(np.random.choice(range(100, 200), size=n_initial, replace=False))
learner_list = [ActiveLearner(
                        predictor=GaussianProcessRegressor(kernel),
                        X_initial=X[idx].reshape(-1, 1), y_initial=y[idx].reshape(-1, 1)
                )
                for idx in initial_idx]

# query strategy for regression
def ensemble_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

# initializing the Committee
committee = CommitteeRegressor(
    learner_list=learner_list,
    query_strategy=ensemble_regression_std
)

for learner in committee:
    print(learner)