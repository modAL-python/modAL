from modAL.density import information_density, similarize_distance
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs

X, y = make_blobs(n_features=2, n_samples=10, centers=3, random_state=0, cluster_std=0.7)

cosine_density = information_density(X)
euclidean_density = information_density(X, similarize_distance(euclidean))
