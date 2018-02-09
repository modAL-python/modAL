import numpy as np


def make_linear_combination(*functions, weights=None):
    """
    Takes the given functions and makes a function which returns the linear combination
    of the output of original functions. It works well with functions returning numpy
    arrays of the same shape.

    Parameters
    ----------
    *functions: base functions for the linear combination
        The functions shall have the same argument and if they return numpy arrays,
        the returned arrays shall have the same shape.

    weights: array-like, length shall match the number of functions given
        Coefficients of the functions in the linear combination. The i-th given function
        will be multiplied with with weights[i]

    Returns
    -------
    linear_combination: function
        A function which returns the linear combination of the given functions output.
    """

    if weights is None:
        weights = np.ones(shape=(len(functions)))
    else:
        assert len(functions) == len(weights), 'the length of weights must be the ' \
                                               'same as the number of given functions'

    def linear_combination(*args, **kwargs):
        return np.sum([weights[i]*functions[i](*args, **kwargs)
                       for i in range(len(weights))], axis=0)

    return linear_combination


def make_product(*functions, exponents=None):
    """
    Takes the given functions and makes a function which returns the product of the output
    of original functions. It works well with functions returning numpy arrays of the same
    shape.

    Parameters
    ----------
    *functions: base functions for the product
        The functions shall have the same argument and if they return numpy arrays,
        the returned arrays shall have the same shape.

    exponents: array-like, length shall match the number of functions given
        Exponents of the functions in the product. The i-th given function in the product
        will be raised to the power of exponents[i]

    Returns
    -------
    product_function: function
        A function which returns the product function of the given functions output.
    """

    if exponents is None:
        exponents = np.ones(shape=(len(functions)))
    else:
        assert len(functions) == len(exponents), 'the length of exponents must be the ' \
                                                 'same as the number of given functions'

    def product_function(*args, **kwargs):
        return np.prod([functions[i](*args, **kwargs)**exponents[i]
                       for i in range(len(exponents))], axis=0)

    return product_function
