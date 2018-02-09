import numpy as np


def make_linear_combination(*functions, weights=None):
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
    if exponents is None:
        exponents = np.ones(shape=(len(functions)))
    else:
        assert len(functions) == len(exponents), 'the length of exponents must be the ' \
                                                 'same as the number of given functions'

    def product_function(*args, **kwargs):
        return np.prod([functions[i](*args, **kwargs)**exponents[i]
                       for i in range(len(exponents))], axis=0)

    return product_function