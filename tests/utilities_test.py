import unittest
import numpy as np
from collections import namedtuple
from modAL.active_learning.utilities import classifier_uncertainty


Test = namedtuple('Test', ['input', 'output'])


class MockClassifier:
    def __init__(self, predict_proba_return):
        self.predict_proba_return = predict_proba_return

    def predict_proba(self, data):
        return self.predict_proba_return


class TestUtilities(unittest.TestCase):

    def test_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k + 1, l + 1)), (1 - p) * np.ones(shape=(k + 1, )))
                      for k in range(1000) for l in range(10) for p in np.linspace(0, 1, 10))
        for case in test_cases:
            mock_classifier = MockClassifier(case.input)
            np.testing.assert_almost_equal(
                classifier_uncertainty(mock_classifier, np.random.rand(10)),
                case.output)


if __name__ == '__main__':
    unittest.main()