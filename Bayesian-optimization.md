# Bayesian optimization

When a function is expensive to evaluate, or when gradients are not available, optimalizing it requires more sophisticated methods than gradient descent. One such method is Bayesian optimization, which lies close to active learning. In Bayesian optimization, instead of picking queries by maximizing the uncertainty of predictions, function values are evaluated at points where the promise of finding a better value is large. In modAL, these algorithms are implemented with the ```BayesianOptimizer``` class, which is a sibling of ```ActiveLearner```. In the following example, their use is demonstrated on a toy problem.

## The function to be optimized

```python
import numpy as np

# generating the data
X = np.linspace(0, 20, 1000).reshape(-1, 1)
y = np.sin(X)/2 - ((10 - X)**2)/50 + 2
```

## Gaussian processes

```python
# assembling initial training set
X_initial, y_initial = X[150].reshape(1, -1), y[150].reshape(1, -1)

# defining the kernel for the Gaussian process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel=kernel)
```

## Optimizing using Expected improvement

```python
# initializing the optimizer
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI
optimizer = BayesianOptimizer(
    estimator=regressor,
    X_training=X_initial, y_training=y_initial,
    query_strategy=max_EI
)

# Bayesian optimization
for n_queries in range(4):
    query_idx, query_inst = optimizer.query(X)
    optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
```

![](img/bo-EI.png)
![](img/bo-PI.png)
![](img/bo-UCB.png)
