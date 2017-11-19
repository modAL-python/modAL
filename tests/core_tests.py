import unittest
import numpy as np
import modAL.utilities
from collections import namedtuple


Test = namedtuple('Test', ['input', 'output'])


class MockClassifier:
    def __init__(self, predict_proba_return):
        self.predict_proba_return = predict_proba_return

    def predict_proba(self, data):
        return self.predict_proba_return


class TestUtilities(unittest.TestCase):

    def test_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k + 1, l + 1)), (1 - p) * np.ones(shape=(k + 1, )))
                      for k in range(100) for l in range(10) for p in np.linspace(0, 1, 10))
        for case in test_cases:
            mock_classifier = MockClassifier(case.input)
            np.testing.assert_almost_equal(
                modAL.utilities.classifier_uncertainty(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_margin(self):
        test_cases = (Test(p * np.ones(shape=(k + 1, l + 1)), np.zeros(shape=(k + 1,)))
                      for k in range(100) for l in range(10) for p in np.linspace(0, 1, 10))
        for case in test_cases:
            mock_classifier = MockClassifier(case.input)
            np.testing.assert_almost_equal(
                modAL.utilities.classifier_margin(mock_classifier, np.random.rand(10)),
                case.output
            )


if __name__ == '__main__':

    unittest.main()