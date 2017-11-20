import unittest
import numpy as np
import modAL.utilities
from itertools import chain
from collections import namedtuple


Test = namedtuple('Test', ['input', 'output'])


class MockClassifier:
    """
    Mock classifier object for testing. The predict_proba method returns the
    object given for argument predict_proba_return.
    """
    def __init__(self, predict_proba_return=None, calculate_utility_return=None):
        self.predict_proba_return = predict_proba_return
        self.calculate_utility_return = calculate_utility_return

    def predict_proba(self, data):
        return self.predict_proba_return

    def calculate_utility(self, data):
        return self.calculate_utility_return


class TestUtilities(unittest.TestCase):

    def test_uncertainty(self):
        print('Testing modAL.utilities.classifier_uncertainty()...')

        test_cases = (Test(p * np.ones(shape=(k + 1, l + 1)), (1 - p) * np.ones(shape=(k + 1, )))
                      for k in range(100) for l in range(10) for p in np.linspace(0, 1, 10))
        for case in test_cases:
            mock_classifier = MockClassifier(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.utilities.classifier_uncertainty(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_margin(self):
        print('Testing modAL.utilities.classifier_margin()...')

        test_cases_1 = (Test(p * np.ones(shape=(k + 1, l + 1)), np.zeros(shape=(k + 1,)))
                      for k in range(100) for l in range(10) for p in np.linspace(0, 1, 10))
        test_cases_2 = (Test(p * np.tile(np.asarray(range(k+1))+1.0, l+1).reshape(l+1, k+1),
                             p * np.ones(shape=(l+1, ))*int(k!=0))
                        for k in range(10) for l in range(100) for p in np.linspace(0, 1, 11))
        for case in chain(test_cases_1, test_cases_2):
            mock_classifier = MockClassifier(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.utilities.classifier_margin(mock_classifier, np.random.rand(10)),
                case.output
            )


class TestCommittee(unittest.TestCase):

    def test_calculate_utility(self):
        print('Testing Committee.calculate_utility()...')

if __name__ == '__main__':
    unittest.main()