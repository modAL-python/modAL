import random
import unittest
import numpy as np

import mock
import modAL.uncertainty
import modAL.query
import modAL.models
import modAL.utils.validation

from itertools import chain
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier


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
                learner_list_1 = [mock.MockClassifier(classes_=labels) for _ in range(n_learners)]
                learner_list_2 = [mock.MockClassifier(classes_=different_labels) for _ in range(np.random.randint(1, 5))]
                shuffled_learners = random.sample(learner_list_1 + learner_list_2, len(learner_list_1 + learner_list_2))
                self.assertTrue(modAL.utils.validation.check_class_labels(*learner_list_1))
                self.assertFalse(modAL.utils.validation.check_class_labels(*shuffled_learners))


class TestUncertainties(unittest.TestCase):

    def test_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k, l)), (1 - p) * np.ones(shape=(k, )))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        for case in test_cases:
            mock_classifier = mock.MockClassifier(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_uncertainty(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_margin(self):
        test_cases_1 = (Test(p * np.ones(shape=(k, l)), np.zeros(shape=(k,)))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        test_cases_2 = (Test(p * np.tile(np.asarray(range(k))+1.0, l).reshape(l, k),
                             p * np.ones(shape=(l, ))*int(k!=1))
                        for k in range(1, 10) for l in range(1, 100) for p in np.linspace(0, 1, 11))
        for case in chain(test_cases_1, test_cases_2):
            mock_classifier = mock.MockClassifier(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_margin(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_entropy(self):
        for n_samples in range(1, 100):
            for n_classes in range(1, 20):
                proba = np.zeros(shape=(n_samples, n_classes))
                for sample_idx in range(n_samples):
                    proba[sample_idx, np.random.choice(range(n_classes))] = 1.0

                classifier = mock.MockClassifier(predict_proba_return=proba)
                np.testing.assert_equal(
                    modAL.uncertainty.classifier_entropy(classifier, np.random.rand(n_samples, 1)),
                    np.zeros(shape=(n_samples, ))
                )


class TestQueries(unittest.TestCase):

    def test_max_utility(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool):
                utility = np.zeros(n_pool)
                max_idx = np.random.choice(range(n_pool), size=n_instances, replace=False)
                utility[max_idx] = 1.0
                np.testing.assert_equal(
                    np.sort(modAL.query.max_uncertainty(utility, n_instances)),
                    np.sort(max_idx)
                )

    def test_utility_weighted_random(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool):
                utility = np.ones(n_pool)
                query_idx = modAL.query.uncertainty_weighted_random(utility, n_instances)
                # testing for correct number of returned indices
                np.testing.assert_equal(len(query_idx), n_instances)
                # testing for uniqueness of each query index
                np.testing.assert_equal(len(query_idx), len(np.unique(query_idx)))


class TestActiveLearner(unittest.TestCase):

    def test_add_training_data(self):
        for n_samples in range(1, 10):
            for n_features in range(1, 10):
                for n_new_samples in range(1, 10):
                    # testing for valid cases
                    # 1. integer class labels
                    X_initial = np.random.rand(n_samples, n_features)
                    y_initial = np.random.randint(0, 2, size=(n_samples,))
                    X_new = np.random.rand(n_new_samples, n_features)
                    y_new = np.random.randint(0, 2, size=(n_new_samples,))
                    learner = modAL.models.ActiveLearner(
                        predictor=mock.MockClassifier(),
                        X_initial=X_initial, y_initial=y_initial
                    )
                    learner.add_training_data(X_new, y_new)
                    np.testing.assert_almost_equal(
                        learner._X_training,
                        np.vstack((X_initial, X_new))
                    )
                    np.testing.assert_equal(
                        learner._y_training,
                        np.concatenate((y_initial, y_new))
                    )
                    # 2. vector class labels
                    y_initial = np.random.randint(0, 2, size=(n_samples, n_features+1))
                    y_new = np.random.randint(0, 2, size=(n_new_samples, n_features+1))
                    learner = modAL.models.ActiveLearner(
                        predictor=mock.MockClassifier(),
                        X_initial=X_initial, y_initial=y_initial
                    )
                    learner.add_training_data(X_new, y_new)
                    np.testing.assert_equal(
                        learner._y_training,
                        np.concatenate((y_initial, y_new))
                    )

                    # testing for invalid cases
                    # 1. len(X_new) != len(y_new)
                    X_new = np.random.rand(n_new_samples, n_features)
                    y_new = np.random.randint(0, 2, size=(2*n_new_samples,))
                    self.assertRaises(AssertionError, learner.add_training_data, X_new, y_new)
                    # 2. X_new has wrong dimensions
                    X_new = np.random.rand(n_new_samples, 2*n_features)
                    y_new = np.random.randint(0, 2, size=(n_new_samples,))
                    self.assertRaises(ValueError, learner.add_training_data, X_new, y_new)

    def test_calculate_uncertainty(self):
        test_cases = (Test(array, array) for k in range(1, 10) for l in range(1, 10) for array in random_array((k, l), 100))
        for case in test_cases:
            learner = modAL.models.ActiveLearner(mock.MockClassifier(), mock.MockFunction(case.input))
            np.testing.assert_almost_equal(
                learner.calculate_uncertainty(case.input),
                case.output
            )

    def test_predict(self):
        for n_samples in range(1, 100):
            for n_features in range(1, 10):
                X = np.random.rand(n_samples, n_features)
                predict_return = np.random.randint(0, 2, size=(n_samples, ))
                mock_classifier = mock.MockClassifier(predict_return=predict_return)
                learner = modAL.models.ActiveLearner(
                    predictor=mock_classifier
                )
                np.testing.assert_equal(
                    learner.predict(X),
                    predict_return
                )

    def test_predict_proba(self):
        for n_samples in range(1, 100):
            for n_features in range(1, 10):
                X = np.random.rand(n_samples, n_features)
                predict_proba_return = np.random.randint(0, 2, size=(n_samples,))
                mock_classifier = mock.MockClassifier(predict_proba_return=predict_proba_return)
                learner = modAL.models.ActiveLearner(
                    predictor=mock_classifier
                )
                np.testing.assert_equal(
                    learner.predict_proba(X),
                    predict_proba_return
                )

    def test_query(self):
        for n_samples in range(1, 100):
            for n_features in range(1, 10):
                X = np.random.rand(n_samples, n_features)
                query_idx = np.random.randint(0, n_samples)
                mock_query = mock.MockFunction(return_val=query_idx)
                mock_uncertainty = mock.MockFunction(return_val=None)
                learner = modAL.models.ActiveLearner(
                    predictor=None,
                    uncertainty_measure=mock_uncertainty, query_strategy=mock_query
                )
                np.testing.assert_equal(
                    learner.query(X),
                    (query_idx, X[query_idx])
                )

    def test_score(self):
        test_cases = (np.random.rand() for _ in range(10))
        for score_return in test_cases:
            mock_classifier = mock.MockClassifier(score_return=score_return)
            learner = modAL.models.ActiveLearner(mock_classifier, mock.MockFunction(None))
            np.testing.assert_almost_equal(
                learner.score(np.random.rand(5, 2), np.random.rand(5, )),
                score_return
            )

    def test_keras(self):
        pass

    def test_sklearn(self):
        learner = modAL.models.ActiveLearner(
            predictor=RandomForestClassifier(),
            X_initial=np.random.rand(10, 10),
            y_initial=np.random.randint(0, 2, size=(10,))
        )
        learner.fit(np.random.rand(10, 10), np.random.randint(0, 2, size=(10,)))
        learner.predict(np.random.rand(10, 10))
        learner.predict_proba(np.random.rand(10, 10))


class TestCommittee(unittest.TestCase):

    def test_set_classes(self):
        for n_classes in range(1, 10):
            learner_list = [modAL.models.ActiveLearner(predictor=mock.MockClassifier(classes_=np.asarray([idx])))
                            for idx in range(n_classes)]
            committee = modAL.models.Committee(learner_list=learner_list)
            np.testing.assert_equal(
                committee.classes_,
                np.unique(range(n_classes))
            )

    def test_calculate_uncertainty(self):
        for n_learners in range(1, 200):
            utility = np.random.rand(100, n_learners)
            committee = modAL.models.Committee(
                learner_list=[mock.MockActiveLearner(
                                    mock.MockClassifier(classes_=np.asarray([0])),
                                    calculate_utility_return=utility[:, learner_idx].reshape(-1)
                              )
                              for learner_idx in range(n_learners)]
            )
            np.testing.assert_almost_equal(
                committee.calculate_uncertainty(np.random.rand(100, 1)),
                utility
            )

    def test_predict(self):
        for n_learners in range(1, 10):
            for n_instances in range(1, 10):
                prediction = np.random.randint(10, size=(n_instances, n_learners))
                committee = modAL.models.Committee(
                    learner_list=[mock.MockActiveLearner(
                                          mock.MockClassifier(classes_=np.asarray([0])),
                                          predict_return=prediction[:, learner_idx]
                                  )
                                  for learner_idx in range(n_learners)]
                )
                np.testing.assert_equal(
                    committee.vote(np.random.rand(n_instances, 5)),
                    prediction
                )


if __name__ == '__main__':
    unittest.main()
