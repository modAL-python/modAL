import unittest
import numpy as np
import modAL.utils.np_utils
from collections import namedtuple
from itertools import chain


Test = namedtuple('Test', ['input', 'output'])


def random_mask_2D(num, size_k, size_l):
    count = 0
    while count < num:
        count += 1
        mask = np.random.randint(0, 2, size=size_k) == 1
        yield np.repeat(mask, size_l).reshape(size_k, size_l)


class TestUtils(unittest.TestCase):

    def test_max_margin(self):
        # test cases for np.ndarrays
        test_cases_1 = (Test(p * np.tile(np.asarray(range(k)) + 1.0, l).reshape(l, k),
                             p * np.ones(shape=(l,)) * int(k != 1))
                        for k in range(1, 10) for l in range(1, 100) for p in np.linspace(0, 1, 11))
        # test cases for np.ma.masked_arrays
        test_cases_2 = (
            Test(
                np.ma.masked_array(np.tile(np.asarray(range(k)) + 1.0, l).reshape(l, k), mask=mask),
                np.ones(shape=(l,)) * int(k != 1) * (1 - mask[:, 0])
            )
            for k in range(1, 20) for l in range(1, 10) for mask in random_mask_2D(1, l, k)
        )
        for case in chain(test_cases_1, test_cases_2):
            np.testing.assert_almost_equal(
                modAL.utils.np_utils.max_margin(case.input),
                case.output
            )


if __name__ == '__main__':
    unittest.main()
