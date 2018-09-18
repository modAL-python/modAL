import numpy as np
import scipy.sparse as sp


def data_vstack(blocks):
    """
    Stack vertically both sparse and dense arrays.
    """
    if isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif sp.issparse(blocks[0]):
        return sp.vstack(blocks)
    else:
        raise TypeError('%s datatype is not supported' % type(blocks[0]))
