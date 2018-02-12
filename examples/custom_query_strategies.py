import numpy as np
import matplotlib.pyplot as plt

from modAL.utils.combination import make_linear_combination, make_product
from modAL.uncertainty import classifier_uncertainty, classifier_margin
from modAL.models import ActiveLearner
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

centers = np.asarray([[-2, 3], [0.5, 5], [1, 1.5]])
X, y = make_blobs(
    n_features=2, n_samples=1000, random_state=0, cluster_std=0.7,
    centers=centers
)
# visualizing the dataset
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap='viridis', s=50)
    plt.title('The iris dataset')
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
    uncertainty = classifier_uncertainty(learner, X)
    margin = classifier_margin(learner, X)
    lc = linear_combination(learner, X)
    prod = product(learner, X)
    plt.figure(figsize=(18, 14))
    plt.subplot(2, 2, 1)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=uncertainty, cmap='viridis', s=50)
    plt.title('Classifier uncertainty')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=margin, cmap='viridis', s=50)
    plt.title('Classifier margin')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=lc, cmap='viridis', s=50)
    plt.title('1.0*uncertainty + 1.0*margin')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=prod, cmap='viridis', s=50)
    plt.title('(uncertainty**0.5)*(margin**0.5)')
    plt.colorbar()
    plt.show()
