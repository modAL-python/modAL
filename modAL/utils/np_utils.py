import numpy as np
from sklearn.utils import check_array


def max_margin(array, axis=None):
    """
    Calculates the difference between the first and second largest element of an array.
    :param array: numpy.ndarray
    :param axis: axis along which to calculate the max margin
    :return: max margin of array or
    """
    check_array(array, ensure_2d=True)

    margins = np.zeros(shape=(array.shape[0],))

    for elem_idx, elem in enumerate(array):
        first_max = -np.inf
        second_max = -np.inf

        for val in elem:
            if val > first_max:
                second_max = first_max
                first_max = val

        if second_max != -np.inf:
            margins[elem_idx] = first_max - second_max
        else:
            margins[elem_idx] = 0.0

    return margins