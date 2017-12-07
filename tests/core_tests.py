import random
import unittest
import numpy as np
import modAL.utilities
import modAL.models
import modAL.utils.validation
from itertools import chain
from collections import namedtuple
from mock import MockClassifier, MockUtility


Test = namedtuple('Test', ['input', 'output'])


def random_array(shape, n_arrays):
    for _ in range(n_arrays):
        yield np.random.rand(*shape)


class TestUtils(unittest.TestCase):

    def test_check_class_labels(self):
        for n_labels in range(1, 10):
            for n_learners in range(1, 10):
                labels = np.random.randint(10, size=n_labels)
                different_labels = np.random.randint(10, 20, size=np.random.randint(1, 10))
                learner_list_1 = [MockClassifier(classes_=labels) for _ in range(n_learners)]
                learner_list_2 = [MockClassifier(classes_=different_labels) for _ in range(np.random.randint(1, 5))]
                shuffled_learners = random.sample(learner_list_1 + learner_list_2, len(learner_list_1 + learner_list_2))
                self.assertTrue(modAL.utils.validation.check_class_labels(*learner_list_1))
                self.assertFalse(modAL.utils.validation.check_class_labels(*shuffled_learners))


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
        test_cases = (Test(array, array) for k in range(1, 10) for l in range(1, 10) for array in random_array((k, l), 100))
        for case in test_cases:
            mock_classifier = MockClassifier(calculate_utility_return=case.input)
            learner = modAL.models.ActiveLearner(mock_classifier, MockUtility(case.input))
            np.testing.assert_almost_equal(
                learner.calculate_utility(case.input),
                case.output
            )

    def test_query(self):
        pass

    def test_predict(self):
        pass

    def test_predict_proba(self):
        pass

    def test_score(self):
        test_cases = (np.random.rand() for _ in range(10))
        for score_return in test_cases:
            mock_classifier = MockClassifier(score_return=score_return)
            learner = modAL.models.ActiveLearner(mock_classifier, MockUtility(None))
            np.testing.assert_almost_equal(
                learner.score(np.random.rand(5, 2), np.random.rand(5, )),
                score_return
            )

    def test_keras(self):
        pass

    def test_sklearn(self):
        pass


class TestCommittee(unittest.TestCase):

    def test_calculate_utility(self):
        for n_learners in range(1, 200):
            utility = np.random.rand(100, n_learners)
            committee = modAL.models.Committee(
                learner_list=[MockClassifier(calculate_utility_return=utility[:, learner_idx].reshape(-1),)
                              for learner_idx in range(n_learners)],
                voting_function=None
            )
            np.testing.assert_almost_equal(
                committee.calculate_utility(np.random.rand(100, 1)),
                utility
            )

    def test_predict(self):
        for n_learners in range(1, 10):
            for n_instances in range(1, 10):
                prediction = np.random.randint(10, size=(n_instances, n_learners))
                committee = modAL.models.Committee(
                    learner_list=[MockClassifier(predict_return=prediction[:, learner_idx])
                                  for learner_idx in range(n_learners)],
                    voting_function=None
                )
                np.testing.assert_equal(
                    committee.predict(np.random.rand(n_instances, 5)),
                    prediction
                )

if __name__ == '__main__':
    unittest.main()
