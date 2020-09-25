from typing import Union, List, Sequence
from itertools import chain

import numpy as np
import pandas as pd
import scipy.sparse as sp


modALinput = Union[list, np.ndarray, sp.csr_matrix, pd.DataFrame]


def data_vstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack vertically sparse/dense arrays and pandas data frames.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of vertically stacked elements.
    """
    if isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        return list(chain(blocks))
    elif sp.issparse(blocks[0]):
        return sp.vstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        return blocks[0].append(blocks[1])
    else:
        try:
            return np.concatenate(blocks)
        except:
            raise TypeError('%s datatype is not supported' % type(blocks[0]))


def data_hstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack horizontally both sparse and dense arrays

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of horizontally stacked elements.
    """
    # use sparse representation if any of the blocks do
    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)

    try:
        return np.hstack(blocks)
    except:
        raise TypeError('%s datatype is not supported' % type(blocks[0]))


def retrieve_rows(X: modALinput,
                  I: Union[int, List[int], np.ndarray]) -> Union[sp.csc_matrix, np.ndarray, pd.DataFrame]:
    """
    Returns the rows I from the data set X
    """
    if isinstance(X, pd.DataFrame):
        return X.iloc[I]

    return X[I]

def drop_rows(X: modALinput,
              I: Union[int, List[int], np.ndarray]) -> Union[sp.csc_matrix, np.ndarray, pd.DataFrame]:
    if isinstance(X, pd.DataFrame):
        return X.drop(I, axis=0)

    return np.delete(X, I, axis=0)

def enumerate_data(X: modALinput):
    if isinstance(X, pd.DataFrame):
        return X.iterrows()

    return enumerate(X)
