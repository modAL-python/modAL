import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from modAL.models import ActiveLearner, Committee
from modAL.utilities import classifier_uncertainty

# loading the iris dataset
iris = load_iris()
# visualizing the classes
with plt.style.context('seaborn-white'):
    pca = PCA(n_components=2).fit_transform(iris['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=iris['target'], cmap='viridis')
    plt.title('The iris dataset')
    plt.show()

# generate the pool
pool_data = deepcopy(iris['data'])
pool_labels = deepcopy(iris['target'])

# initializing Committee members
n_members = 2
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 5
    train_idx = np.random.choice(range(pool_data.shape[0]), size=n_initial, replace=False)
    X_train = pool_data[train_idx]
    y_train = pool_labels[train_idx]

    # creating a reduced copy of the data with the known instances removed
    pool_data = np.delete(pool_data, train_idx, axis=0)
    pool_labels = np.delete(pool_labels, train_idx)

    # initializing learner
    learner = ActiveLearner(
        predictor=MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000),
        utility_function=classifier_uncertainty,
        training_data=X_train, training_labels=y_train
    )
    learner_list.append(learner)

# assembling the Committee
committee = Committee(
    learner_list=learner_list, voting_function=None
)
