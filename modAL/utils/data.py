import sys
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import torch
except ImportError:
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

    if not blocks:
        return blocks

    types = {type(block) for block in blocks}

    if any([sp.issparse(b) for b in blocks]):
        return sp.vstack(blocks)
    elif types - {pd.DataFrame, pd.Series} == set():
        def _block_to_df(block):
            if isinstance(block, pd.DataFrame):
                return block
            elif isinstance(block, pd.Series):
                # interpret series as a row
                return block.to_frame().T
            else:
                raise TypeError(f"Expected DataFrame or Series but encountered {type(block)}")

        return pd.concat([_block_to_df(block) for block in blocks])
    elif types == {np.ndarray}:
        return np.concatenate(blocks)
    elif types == {list}:
        return np.concatenate(blocks).tolist()

    if 'torch' in sys.modules and all(torch.is_tensor(block) for block in blocks):
        return torch.cat(blocks)

    raise TypeError("%s datatype(s) not supported" % types)


def data_hstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack horizontally sparse/dense arrays and pandas data frames.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of horizontally stacked elements.
    """

    if not blocks:
        return blocks

    types = {type(block) for block in blocks}

    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)
    elif types == {pd.DataFrame}:
        pd.concat(blocks, axis=1)
    elif types == {np.ndarray}:
        return np.hstack(blocks)
    elif types == {list}:
        return np.hstack(blocks).tolist()

    if 'torch' in sys.modules and torch.is_tensor(blocks[0]):
        return torch.cat(blocks, dim=1)

    raise TypeError("%s datatype(s) not supported" % types)


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
    X: modALinput, I: Union[int, List[int], np.ndarray]  # noqa: E741
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
    except (KeyError, IndexError, TypeError):
        pass

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
    X: modALinput, I: Union[int, List[int], np.ndarray]  # noqa: E741
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
    if isinstance(X, list):
        return np.array(X).shape
    elif hasattr(X, "shape"):  # scipy.sparse, torch, pandas and numpy all support .shape
        return X.shape

    raise TypeError("%s datatype is not supported" % type(X))
