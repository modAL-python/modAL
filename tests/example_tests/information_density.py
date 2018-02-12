import matplotlib.pyplot as plt

from modAL.density import similarize_distance, information_density
from sklearn.datasets import make_blobs
from scipy.spatial.distance import euclidean

X, y = make_blobs(n_features=2, n_samples=10, centers=3, random_state=0, cluster_std=0.7)

cosine_density = information_density(X)
euclidean_density = information_density(X, similarize_distance(euclidean))
