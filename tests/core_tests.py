import unittest
import numpy as np
import modAL.utilities
import modAL.models
from itertools import chain
from collections import namedtuple


Test = namedtuple('Test', ['input', 'output'])


class MockClassifier:
    """
    Mock classifier object for testing. The predict_proba method returns the
    object given for argument predict_proba_return.
    """
    def __init__(
            self, predict_proba_return=None, calculate_utility_return=None, predict_return=None
    ):
        self.predict_proba_return = predict_proba_return
        self.calculate_utility_return = calculate_utility_return
        self.predict_return = None

    def predict_proba(self, data):
        return self.predict_proba_return

    def predict(self, data):
        return self.predict_return

    def calculate_utility(self, data):
        return self.calculate_utility_return


class TestUtilities(unittest.TestCase):

    def test_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k, l)), (1 - p) * np.ones(shape=(k, )))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        for case in test_cases:
            mock_classifier = MockClassifier(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.utilities.classifier_uncertainty(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_margin(self):
        test_cases_1 = (Test(p * np.ones(shape=(k, l)), np.zeros(shape=(k,)))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        test_cases_2 = (Test(p * np.tile(np.asarray(range(k))+1.0, l).reshape(l, k),
                             p * np.ones(shape=(l, ))*int(k!=1))
                        for k in range(1, 10) for l in range(1, 100) for p in np.linspace(0, 1, 11))
        for case in chain(test_cases_1, test_cases_2):
            mock_classifier = MockClassifier(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.utilities.classifier_margin(mock_classifier, np.random.rand(10)),
                case.output
            )


class TestCommittee(unittest.TestCase):

    def test_calculate_utility(self):
        for n_learners in range(1, 200):
            utility = np.random.rand(100, n_learners)
            committee = modAL.models.Committee(
                learner_list=[MockClassifier(calculate_utility_return=utility[:, learner_idx])
                              for learner_idx in range(n_learners)]
            )
            np.testing.assert_almost_equal(
                committee.calculate_utility(np.random.rand(100, 1)),
                utility
            )


if __name__ == '__main__':
    unittest.main()
