"""
In this example the performance of the active classification is demonstrated on the iris dataset.
For more information on the iris dataset, see https://en.wikipedia.org/wiki/Iris_flower_data_set
For its scikit-learn interface, see http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from modAL.models import ActiveLearner
from modAL.utilities import classifier_uncertainty

# loading the iris dataset
iris = load_iris()
# visualizing the classes
with plt.style.context('seaborn-white'):
    pca = PCA(n_components=2).fit_transform(iris['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=iris['target'], cmap='viridis')
    plt.title('The iris dataset')
    plt.show()

# initial training data
n_initial = 10
train_idx = np.random.choice(range(iris['data'].shape[0]), size=n_initial, replace=False)
X_train = iris['data'][train_idx]
y_train = iris['target'][train_idx]
# creating a reduced copy of the data with the known instances removed
pool_data = np.delete(iris['data'], train_idx, axis=0)
pool_labels = np.delete(iris['target'], train_idx)

# active learning
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
learner = ActiveLearner(
    predictor=mlp_classifier, utility_function=classifier_uncertainty,
    training_data=X_train, training_labels=y_train
)

n_queries = 20
for idx in range(n_queries):
    query_idx, query_instance = learner.query(pool_data)
    learner.add_and_retrain(
        new_data=pool_data[query_idx].reshape(1, -1),
        new_label=pool_labels[query_idx].reshape(-1, )
    )
    # remove queried instance from pool
    pool_data = np.delete(pool_data, query_idx, axis=0)
    pool_labels = np.delete(pool_labels, query_idx)

with plt.style.context('seaborn-white'):
    prediction = learner.predict(iris['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis')
    plt.title('Predictions after %i queries' % n_queries)
    plt.show()
