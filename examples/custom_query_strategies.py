"""
=============================
Template for query strategies
=============================

The first two arguments of a query strategy function is always the estimator and the pool
of instances to be queried from. Additional arguments are accepted as keyword arguments.
A valid query strategy function always returns indices of the queried
instances.

def custom_query_strategy(classifier, X, a_keyword_argument=42):
    # measure the utility of each instance in the pool
    utility = utility_measure(classifier, X)

    # select and return the indices of the instances to be queried
    return select_instances(utility)


This function can be used in the active learning workflow.

learner = ActiveLearner(classifier, query_strategy=custom_query_strategy)
query_idx, query_instance = learner.query(X_pool, a_keyword_argument=42*42)

In this example, we will construct a custom query function by combining classifier uncertainty
and classifier margin.
"""

import matplotlib.pyplot as plt
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_margin, classifier_uncertainty
from modAL.utils.combination import make_linear_combination, make_product
from modAL.utils.selection import multi_argmax
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# generating the data
centers = np.asarray([[-2, 3], [0.5, 5], [1, 1.5]])
X, y = make_blobs(
    n_features=2, n_samples=1000, random_state=0, cluster_std=0.7,
    centers=centers
)
# visualizing the dataset
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap='viridis', s=50)
    plt.title('The dataset')
    plt.show()

# initial training data
initial_idx = np.random.choice(range(len(X)), size=20)
X_training, y_training = X[initial_idx], y[initial_idx]

# initializing the learner
learner = ActiveLearner(
    estimator=GaussianProcessClassifier(1.0 * RBF(1.0)),
    X_training=X_training, y_training=y_training
)

# creating new utility measures by linear combination and product
# linear_combination will return 1.0*classifier_uncertainty + 1.0*classifier_margin
linear_combination = make_linear_combination(
    classifier_uncertainty, classifier_margin,
    weights=[1.0, 1.0]
)
# product will return (classifier_uncertainty**0.5)*(classifier_margin**0.1)
product = make_product(
    classifier_uncertainty, classifier_margin,
    exponents=[0.5, 0.1]
)

# visualizing the different utility metrics
with plt.style.context('seaborn-white'):
    utilities = [
        (1, classifier_uncertainty(learner, X), 'Classifier uncertainty'),
        (2, classifier_margin(learner, X), 'Classifier margin'),
        (3, linear_combination(learner, X), '1.0*uncertainty + 1.0*margin'),
        (4, product(learner, X), '(uncertainty**0.5)*(margin**0.5)')
    ]

    plt.figure(figsize=(18, 14))
    for idx, utility, title in utilities:
        plt.subplot(2, 2, idx)
        plt.scatter(x=X[:, 0], y=X[:, 1], c=utility, cmap='viridis', s=50)
        plt.title(title)
        plt.colorbar()

    plt.show()


# defining the custom query strategy, which uses the linear combination of
# classifier uncertainty and classifier margin
def custom_query_strategy(classifier, X, n_instances=1):
    utility = linear_combination(classifier, X)
    return multi_argmax(utility, n_instances=n_instances)

custom_query_learner = ActiveLearner(
    estimator=GaussianProcessClassifier(1.0 * RBF(1.0)),
    query_strategy=custom_query_strategy,
    X_training=X_training, y_training=y_training
)

# pool-based sampling
n_queries = 20
for idx in range(n_queries):
    query_idx, query_instance = custom_query_learner.query(X, n_instances=2)
    custom_query_learner.teach(
        X=X[query_idx].reshape(-1, 2),
        y=y[query_idx].reshape(-1, )
    )
