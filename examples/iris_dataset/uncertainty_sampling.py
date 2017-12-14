"""
In this example the performance of the active classification is demonstrated on the iris dataset.
For more information on the iris dataset, see https://en.wikipedia.org/wiki/Iris_flower_data_set
For its scikit-learn interface, see http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

# loading the iris dataset
iris = load_iris()
# visualizing the classes
with plt.style.context('seaborn-white'):
    pca = PCA(n_components=2).fit_transform(iris['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=iris['target'], cmap='viridis')
    plt.title('The iris dataset')
    plt.show()

# initial training data
train_idx = [0, 50, 100]
X_train = iris['data'][train_idx]
y_train = iris['target'][train_idx]
# creating a reduced copy of the data with the known instances removed
pool_data = np.delete(iris['data'], train_idx, axis=0)
pool_labels = np.delete(iris['target'], train_idx)

# active learning
rfc = RandomForestClassifier()
learner = ActiveLearner(
    predictor=rfc,
    X_initial=X_train, y_initial=y_train
)

print('Accuracy before active learning: %f' % learner.score(iris['data'], iris['target']))

n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = learner.query(pool_data)
    learner.teach(
        X=pool_data[query_idx].reshape(1, -1),
        y=pool_labels[query_idx].reshape(1, )
    )
    # remove queried instance from pool
    pool_data = np.delete(pool_data, query_idx, axis=0)
    pool_labels = np.delete(pool_labels, query_idx)
    print('Accuracy after query no. %d: %f' % (idx+1, learner.score(iris['data'], iris['target'])))

with plt.style.context('seaborn-white'):
    prediction = learner.predict(iris['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis')
    plt.title('Classification accuracy after %i queries: %f' % (n_queries, learner.score(iris['data'], iris['target'])))
    plt.show()
