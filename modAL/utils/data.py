from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import torch
except:
    pass


modALinput = Union[sp.csr_matrix, pd.DataFrame, np.ndarray, list]


def data_vstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack vertically sparse/dense arrays and pandas data frames.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of vertically stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.vstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        return blocks[0].append(blocks[1:])
    elif isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        return np.concatenate(blocks).tolist()

    try:
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks)
    except:
        pass

    raise TypeError("%s datatype is not supported" % type(blocks[0]))


def data_hstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack horizontally sparse/dense arrays and pandas data frames.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of horizontally stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        pd.concat(blocks, axis=1)
    elif isinstance(blocks[0], np.ndarray):
        return np.hstack(blocks)
    elif isinstance(blocks[0], list):
        return np.hstack(blocks).tolist()

    try:
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks, dim=1)
    except:
        pass

    TypeError("%s datatype is not supported" % type(blocks[0]))


def add_row(X: modALinput, row: modALinput):
    """
    Returns X' =

    [X

    row]    """
    if isinstance(X, np.ndarray):
        return np.vstack((X, row))
    elif isinstance(X, list):
        return np.vstack((X, row)).tolist()

    # data_vstack readily supports stacking of matrix as first argument
    # and row as second for the other data types
    return data_vstack([X, row])


def retrieve_rows(
    X: modALinput, I: Union[int, List[int], np.ndarray]
) -> Union[sp.csc_matrix, np.ndarray, pd.DataFrame]:
    """
    Returns the rows I from the data set X

    For a single index, the result is as follows:
    * 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame
    * row in case of list or numpy format
    """

    try:
        return X[I]
    except:
        if sp.issparse(X):
            # Out of the sparse matrix formats (sp.csc_matrix, sp.csr_matrix, sp.bsr_matrix,
            # sp.lil_matrix, sp.dok_matrix, sp.coo_matrix, sp.dia_matrix), only sp.bsr_matrix, sp.coo_matrix
            # and sp.dia_matrix don't support indexing and need to be converted to a sparse format
            # that does support indexing. It seems conversion to CSR is currently most efficient.

            sp_format = X.getformat()
            return X.tocsr()[I].asformat(sp_format)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[I]
        elif isinstance(X, list):
            return np.array(X)[I].tolist()
        elif isinstance(X, dict):
            X_return = {}
            for key, value in X.items():
                X_return[key] = retrieve_rows(value, I)
            return X_return

    raise TypeError("%s datatype is not supported" % type(X))


def drop_rows(
    X: modALinput, I: Union[int, List[int], np.ndarray]
) -> Union[sp.csc_matrix, np.ndarray, pd.DataFrame]:
    """
    Returns X without the row(s) at index/indices I
    """
    if sp.issparse(X):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[I] = False
        return retrieve_rows(X, mask)
    elif isinstance(X, pd.DataFrame):
        return X.drop(I, axis=0)
    elif isinstance(X, np.ndarray):
        return np.delete(X, I, axis=0)
    elif isinstance(X, list):
        return np.delete(X, I, axis=0).tolist()

    try:
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks)
    except:
        X[[True if row not in I else False for row in range(X.size(0))]]

    raise TypeError("%s datatype is not supported" % type(X))


def enumerate_data(X: modALinput):
    """
    for i, x in enumerate_data(X):

    Depending on the data type of X, returns:

    * A 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame X
    * row in case of list or numpy format
    """
    if sp.issparse(X):
        return enumerate(X.tocsr())
    elif isinstance(X, pd.DataFrame):
        return X.iterrows()
    elif isinstance(X, np.ndarray) or isinstance(X, list):
        # numpy arrays and lists can readily be enumerated
        return enumerate(X)

    raise TypeError("%s datatype is not supported" % type(X))


def data_shape(X: modALinput):
    """
    Returns the shape of the data set X
    """
    try:
        # scipy.sparse, torch, pandas and numpy all support .shape
        return X.shape
    except:
        if isinstance(X, list):
            return np.array(X).shape

    raise TypeError("%s datatype is not supported" % type(X))
