import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier

r = 0.5
p = 0.5
h = 0.01

X = np.random.rand(200, 2)
X = np.array([x for x in X if np.linalg.norm(x) > r and np.linalg.norm(x-1) > r])
X_b = np.array([x for x in X if np.linalg.norm(x - np.array([0, 1])) < r and np.random.rand() < p])
X_r = np.array([x for x in X if np.linalg.norm(x - np.array([1, 0])) < r and np.random.rand() < p])
y = np.array([0 for _ in X_b] + [1 for _ in X_r])

gpc = GaussianProcessClassifier()
gpc.fit(np.concatenate((X_b, X_r), axis=0), y)

x0_min, x0_max = X[:, 0].min() - .1, X[:, 0].max() + .1
x1_min, x1_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z = z.reshape(xx.shape)

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, 1-z, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], s=50, c='k', label='Unlabeled')
    plt.scatter(X_b[:, 0], X_b[:, 1], s=100, c='b', edgecolors='k', label='Class 1')
    plt.scatter(X_r[:, 0], X_r[:, 1], s=100, c='r', edgecolors='k', label='Class 2')
    plt.legend()
    plt.title('Classification probabilities')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('./motivating-example.png', dpi=300)
    plt.close('all')

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], s=50, c='k', label='Unlabeled')
    plt.scatter(X_b[:, 0], X_b[:, 1], s=100, c='b', edgecolors='k', label='Class 1')
    plt.scatter(X_r[:, 0], X_r[:, 1], s=100, c='r', edgecolors='k', label='Class 2')
    plt.legend()
    plt.title('A classification problem')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('./motivating-example-data.png', dpi=300)
    plt.close('all')