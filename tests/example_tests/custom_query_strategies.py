import numpy as np

from modAL.utils.combination import make_linear_combination, make_product
from modAL.utils.selection import multi_argmax
from modAL.uncertainty import classifier_uncertainty, classifier_margin
from modAL.models import ActiveLearner
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# generating the data
centers = np.asarray([[-2, 3], [0.5, 5], [1, 1.5]])
X, y = make_blobs(
    n_features=2, n_samples=1000, random_state=0, cluster_std=0.7,
    centers=centers
)

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

# defining the custom query strategy, which uses the linear combination of
# classifier uncertainty and classifier margin
def custom_query_strategy(classifier, X, n_instances=1):
    utility = linear_combination(classifier, X)
    query_idx = multi_argmax(utility, n_instances=n_instances)
    return query_idx, X[query_idx]

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
