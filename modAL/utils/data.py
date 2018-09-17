import numpy as np
import scipy.sparse as sp


def data_vstack(blocks):
    """Stack vertically both sparse and dense arrays."""
    if sp.issparse(blocks[0]):
        return sp.vstack(blocks)
    return np.concatenate(blocks)
