"""
========================================
Cluster-based active learning algorithms
========================================
"""

import numpy as np
from sklearn.cluster.hierarchical import AgglomerativeClustering, _hc_cut


class HierarchicalClustering:
    def __init__(self, X, classes, n_batch=1):
        self.classes = classes
        self.cluster = AgglomerativeClustering(compute_full_tree=True)
        self.cluster.fit(X)

    def __call__(self, *args, **kwargs):
        pass

    def compute_errors(self):
        pass
