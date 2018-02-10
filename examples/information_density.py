import matplotlib.pyplot as plt

from modAL.density import similarize_distance, information_density
from sklearn.datasets import make_blobs
from scipy.spatial.distance import euclidean

X, y = make_blobs(n_features=2, n_samples=1000, centers=3, random_state=0, cluster_std=0.7)

cosine_density = information_density(X)
euclidean_density = information_density(X, similarize_distance(euclidean))

# visualizing the cosine and euclidean information density
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=cosine_density, cmap='viridis', s=50)
    plt.title('The cosine information density')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=euclidean_density, cmap='viridis', s=50)
    plt.title('The euclidean information density')
    plt.colorbar()
    plt.show()
