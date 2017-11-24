import unittest
import numpy as np
import modAL.utilities
import modAL.models
from itertools import chain
from collections import namedtuple
from mock import MockClassifier


Test = namedtuple('Test', ['input', 'output'])


def random_array(shape, n_arrays):
    for _ in range(n_arrays):
        yield np.random.rand(*shape)


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


class TestActiveLearner(unittest.TestCase):

    def test_calculate_utility(self):
        test_cases = (Test(array, array) for k in range(1, 10) for array in random_array((k, ), 100))
        for case in test_cases:
            mock_classifier = MockClassifier(calculate_utility_return=case.input)
            learner = modAL.models.ActiveLearner
            np.testing.assert_almost_equal(
                mock_classifier.calculate_utility(case.input),
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
