import random
import unittest
import numpy as np

import mock
import modAL.models
import modAL.uncertainty
import modAL.disagreement
import modAL.density
import modAL.utils.selection
import modAL.utils.validation
import modAL.utils.combination

from copy import deepcopy
from itertools import chain, product
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy, norm
from scipy.special import ndtr


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
                learner_list_1 = [mock.MockEstimator(classes_=labels) for _ in range(n_learners)]
                learner_list_2 = [mock.MockEstimator(classes_=different_labels) for _ in range(np.random.randint(1, 5))]
                shuffled_learners = random.sample(learner_list_1 + learner_list_2, len(learner_list_1 + learner_list_2))
                self.assertTrue(modAL.utils.validation.check_class_labels(*learner_list_1))
                self.assertFalse(modAL.utils.validation.check_class_labels(*shuffled_learners))

    def test_check_class_proba(self):
        for n_labels in range(2, 20):
            # when all classes are known:
            proba = np.random.rand(100, n_labels)
            class_labels = list(range(n_labels))
            np.testing.assert_almost_equal(
                modAL.utils.check_class_proba(proba, known_labels=class_labels, all_labels=class_labels),
                proba
            )
            for unknown_idx in range(n_labels):
                all_labels = list(range(n_labels))
                known_labels = deepcopy(all_labels)
                known_labels.remove(unknown_idx)
                aug_proba = np.insert(proba[:, known_labels], unknown_idx, np.zeros(len(proba)), axis=1)
                np.testing.assert_almost_equal(
                    modAL.utils.check_class_proba(proba[:, known_labels], known_labels=known_labels, all_labels=all_labels),
                    aug_proba
                )

    def test_linear_combination(self):
        for n_dim in range(1, 5):
            shape = tuple([10] + [2 for _ in range(n_dim-1)])
            X_in = np.ones(shape=shape)
            for n_functions in range(1, 10):
                functions = [(lambda x: x) for _ in range(n_functions)]
                # linear combination without weights
                linear_combination = modAL.utils.combination.make_linear_combination(*functions)
                np.testing.assert_almost_equal(
                    linear_combination(X_in),
                    n_functions*X_in
                )

                # linear combination with weights
                weights = np.random.rand(n_functions)
                weighted_linear_combination = modAL.utils.combination.make_linear_combination(*functions, weights=weights)
                np.testing.assert_almost_equal(
                    weighted_linear_combination(X_in),
                    np.sum(weights) * X_in
                )

    def test_product(self):
        for n_dim in range(1, 5):
            shape = tuple([10] + [2 for _ in range(n_dim-1)])
            X_in = 2*np.ones(shape=shape)
            for n_functions in range(1, 10):
                functions = [(lambda x: x) for _ in range(n_functions)]
                # linear combination without weights
                product = modAL.utils.combination.make_product(*functions)
                np.testing.assert_almost_equal(
                    product(X_in),
                    X_in**n_functions
                )

                # linear combination with weights
                exponents = np.random.rand(n_functions)
                exp_product = modAL.utils.combination.make_product(*functions, exponents=exponents)
                np.testing.assert_almost_equal(
                    exp_product(X_in),
                    np.prod([X_in**exponent for exponent in exponents], axis=0)
                )

    def test_make_query_strategy(self):
        query_strategy = modAL.utils.combination.make_query_strategy(
            utility_measure=modAL.uncertainty.classifier_uncertainty,
            selector=modAL.utils.selection.multi_argmax
        )

        for n_samples in range(1, 10):
            for n_classes in range(1, 10):
                proba = np.random.rand(n_samples, n_classes)
                proba = proba/np.sum(proba, axis=1).reshape(n_samples, 1)
                X = np.random.rand(n_samples, 3)

                learner = modAL.models.ActiveLearner(
                    estimator=mock.MockEstimator(predict_proba_return=proba)
                )

                query_1 = query_strategy(learner, X)
                query_2 = modAL.uncertainty.uncertainty_sampling(learner, X)

                np.testing.assert_equal(query_1[0], query_2[0])
                np.testing.assert_almost_equal(query_1[1], query_2[1])


class TestAcquisitionFunctions(unittest.TestCase):
    def test_PI(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, )
            std = np.random.rand(n_samples, )
            tradeoff = np.random.rand()
            max_val = np.random.rand()

            mock_estimator = mock.MockEstimator(
                predict_return=(mean, std)
            )
            
            optimizer = modAL.models.BayesianOptimizer(estimator=mock_estimator)
            optimizer._set_max([max_val])

            np.testing.assert_almost_equal(
                ndtr((mean - max_val - tradeoff)/std),
                modAL.acquisition.PI(optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

    def test_EI(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, )
            std = np.random.rand(n_samples, )
            tradeoff = np.random.rand()
            max_val = np.random.rand()

            mock_estimator = mock.MockEstimator(
                predict_return=(mean, std)
            )

            optimizer = modAL.models.BayesianOptimizer(estimator=mock_estimator)
            optimizer._set_max([max_val])

            true_EI = (mean - optimizer.max_val - tradeoff) * ndtr((mean - optimizer.max_val - tradeoff)/std)\
                      + std * norm.pdf((mean - optimizer.max_val - tradeoff)/std)

            np.testing.assert_almost_equal(
                true_EI,
                modAL.acquisition.EI(optimizer, np.random.rand(n_samples, 2), tradeoff)
            )


class TestUncertainties(unittest.TestCase):

    def test_classifier_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k, l)), (1 - p) * np.ones(shape=(k, )))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        for case in test_cases:
            mock_classifier = mock.MockEstimator(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_uncertainty(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_classifier_margin(self):
        test_cases_1 = (Test(p * np.ones(shape=(k, l)), np.zeros(shape=(k,)))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        test_cases_2 = (Test(p * np.tile(np.asarray(range(k))+1.0, l).reshape(l, k),
                             p * np.ones(shape=(l, ))*int(k!=1))
                        for k in range(1, 10) for l in range(1, 100) for p in np.linspace(0, 1, 11))
        for case in chain(test_cases_1, test_cases_2):
            mock_classifier = mock.MockEstimator(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_margin(mock_classifier, np.random.rand(10)),
                case.output
            )

    def test_classifier_entropy(self):
        for n_samples in range(1, 100):
            for n_classes in range(1, 20):
                proba = np.zeros(shape=(n_samples, n_classes))
                for sample_idx in range(n_samples):
                    proba[sample_idx, np.random.choice(range(n_classes))] = 1.0

                classifier = mock.MockEstimator(predict_proba_return=proba)
                np.testing.assert_equal(
                    modAL.uncertainty.classifier_entropy(classifier, np.random.rand(n_samples, 1)),
                    np.zeros(shape=(n_samples, ))
                )

    def test_uncertainty_sampling(self):
        for n_samples in range(1, 10):
            for n_classes in range(1, 10):
                max_proba = np.zeros(n_classes)
                for true_query_idx in range(n_samples):
                    predict_proba = np.random.rand(n_samples, n_classes)
                    predict_proba[true_query_idx] = max_proba
                    classifier = mock.MockEstimator(predict_proba_return=predict_proba)
                    query_idx, query_instance = modAL.uncertainty.uncertainty_sampling(
                        classifier, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_array_equal(query_idx, true_query_idx)

    def test_margin_sampling(self):
        for n_samples in range(1, 10):
            for n_classes in range(1, 10):
                max_proba = np.zeros(n_classes)
                for true_query_idx in range(n_samples):
                    predict_proba = np.random.rand(n_samples, n_classes)
                    predict_proba[true_query_idx] = max_proba
                    classifier = mock.MockEstimator(predict_proba_return=predict_proba)
                    query_idx, query_instance = modAL.uncertainty.uncertainty_sampling(
                        classifier, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_array_equal(query_idx, true_query_idx)

    def test_entropy_sampling(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                max_proba = np.ones(n_classes)/n_classes
                for true_query_idx in range(n_samples):
                    predict_proba = np.zeros(shape=(n_samples, n_classes))
                    predict_proba[:, 0] = 1.0
                    predict_proba[true_query_idx] = max_proba
                    classifier = mock.MockEstimator(predict_proba_return=predict_proba)
                    query_idx, query_instance = modAL.uncertainty.uncertainty_sampling(
                        classifier, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_array_equal(query_idx, true_query_idx)


class TestDensity(unittest.TestCase):

    def test_similarize_distance(self):
        from scipy.spatial.distance import cosine
        sim = modAL.density.similarize_distance(cosine)
        for _ in range(100):
            for n_dim in range(1, 10):
                X_1, X_2 = np.random.rand(n_dim), np.random.rand(n_dim)
                np.testing.assert_almost_equal(
                    sim(X_1, X_2),
                    1/(1 + cosine(X_1, X_2))
                )

    def test_information_density(self):
        for n_samples in range(1, 10):
            for n_dim in range(1, 10):
                X_pool = np.random.rand(n_samples, n_dim)
                similarities = modAL.density.information_density(X_pool)
                np.testing.assert_equal(len(similarities), n_samples)


class TestDisagreements(unittest.TestCase):

    def test_vote_entropy(self):
        for n_samples in range(1, 10):
            for n_classes in range(1, 10):
                for true_query_idx in range(n_samples):
                    vote_return = np.zeros(shape=(n_samples, n_classes), dtype=np.int16)
                    vote_return[true_query_idx] = np.asarray(range(n_classes), dtype=np.int16)
                    committee = mock.MockCommittee(classes_=np.asarray(range(n_classes)), vote_return=vote_return)
                    vote_entr = modAL.disagreement.vote_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    true_entropy = np.zeros(shape=(n_samples, ))
                    true_entropy[true_query_idx] = entropy(np.ones(n_classes)/n_classes)
                    np.testing.assert_array_almost_equal(vote_entr, true_entropy)

    def test_consensus_entropy(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                for true_query_idx in range(n_samples):
                    proba = np.zeros(shape=(n_samples, n_classes))
                    proba[:, 0] = 1.0
                    proba[true_query_idx] = np.ones(n_classes)/n_classes
                    committee = mock.MockCommittee(predict_proba_return=proba)
                    uncertainty_entr = modAL.disagreement.consensus_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    true_entropy = np.zeros(shape=(n_samples,))
                    true_entropy[true_query_idx] = entropy(np.ones(n_classes) / n_classes)
                    np.testing.assert_array_almost_equal(uncertainty_entr, true_entropy)

    def test_KL_max_disagreement(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                for n_learners in range (2, 10):
                    vote_proba = np.zeros(shape=(n_samples, n_learners, n_classes))
                    vote_proba[:, :, 0] = 1.0
                    committee = mock.MockCommittee(
                        n_learners=n_learners, classes_=range(n_classes),
                        vote_proba_return=vote_proba
                    )

                    true_KL_disagreement = np.zeros(shape=(n_samples, ))

                    try:
                        np.testing.assert_array_almost_equal(
                            true_KL_disagreement,
                            modAL.disagreement.KL_max_disagreement(committee, np.random.rand(n_samples, 1))
                        )
                    except:
                        modAL.disagreement.KL_max_disagreement(committee, np.random.rand(n_samples, 1))


class TestQueries(unittest.TestCase):

    def test_multi_argmax(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool):
                utility = np.zeros(n_pool)
                max_idx = np.random.choice(range(n_pool), size=n_instances, replace=False)
                utility[max_idx] = 1.0
                np.testing.assert_equal(
                    np.sort(modAL.utils.selection.multi_argmax(utility, n_instances)),
                    np.sort(max_idx)
                )

    def test_weighted_random(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool):
                utility = np.ones(n_pool)
                query_idx = modAL.utils.selection.weighted_random(utility, n_instances)
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
                        estimator=mock.MockEstimator(),
                        X_training=X_initial, y_training=y_initial
                    )
                    learner._add_training_data(X_new, y_new)
                    np.testing.assert_almost_equal(
                        learner.X_training,
                        np.vstack((X_initial, X_new))
                    )
                    np.testing.assert_equal(
                        learner.y_training,
                        np.concatenate((y_initial, y_new))
                    )
                    # 2. vector class labels
                    y_initial = np.random.randint(0, 2, size=(n_samples, n_features+1))
                    y_new = np.random.randint(0, 2, size=(n_new_samples, n_features+1))
                    learner = modAL.models.ActiveLearner(
                        estimator=mock.MockEstimator(),
                        X_training=X_initial, y_training=y_initial
                    )
                    learner._add_training_data(X_new, y_new)
                    np.testing.assert_equal(
                        learner.y_training,
                        np.concatenate((y_initial, y_new))
                    )

                    # testing for invalid cases
                    # 1. len(X_new) != len(y_new)
                    X_new = np.random.rand(n_new_samples, n_features)
                    y_new = np.random.randint(0, 2, size=(2*n_new_samples,))
                    self.assertRaises(AssertionError, learner._add_training_data, X_new, y_new)
                    # 2. X_new has wrong dimensions
                    X_new = np.random.rand(n_new_samples, 2*n_features)
                    y_new = np.random.randint(0, 2, size=(n_new_samples,))
                    self.assertRaises(ValueError, learner._add_training_data, X_new, y_new)

    def test_predict(self):
        for n_samples in range(1, 100):
            for n_features in range(1, 10):
                X = np.random.rand(n_samples, n_features)
                predict_return = np.random.randint(0, 2, size=(n_samples, ))
                mock_classifier = mock.MockEstimator(predict_return=predict_return)
                learner = modAL.models.ActiveLearner(
                    estimator=mock_classifier
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
                mock_classifier = mock.MockEstimator(predict_proba_return=predict_proba_return)
                learner = modAL.models.ActiveLearner(
                    estimator=mock_classifier
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
                mock_query = mock.MockFunction(return_val=(query_idx, X[query_idx]))
                learner = modAL.models.ActiveLearner(
                    estimator=None,
                    query_strategy=mock_query
                )
                np.testing.assert_equal(
                    learner.query(X),
                    (query_idx, X[query_idx])
                )

    def test_score(self):
        test_cases = (np.random.rand() for _ in range(10))
        for score_return in test_cases:
            mock_classifier = mock.MockEstimator(score_return=score_return)
            learner = modAL.models.ActiveLearner(mock_classifier, mock.MockFunction(None))
            np.testing.assert_almost_equal(
                learner.score(np.random.rand(5, 2), np.random.rand(5, )),
                score_return
            )

    def test_teach(self):
        X_training = np.random.rand(10, 2)
        y_training = np.random.randint(0, 2, size=10)

        for bootstrap, only_new in product([True, False], [True, False]):
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 2)
                y = np.random.randint(0, 2, size=n_samples)

                learner = modAL.models.ActiveLearner(
                    X_training=X_training, y_training=y_training,
                    estimator=mock.MockEstimator()
                )

                learner.teach(X, y, bootstrap=bootstrap, only_new=only_new)

    def test_keras(self):
        pass

    def test_sklearn(self):
        learner = modAL.models.ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=np.random.rand(10, 10),
            y_training=np.random.randint(0, 2, size=(10,))
        )
        learner.fit(np.random.rand(10, 10), np.random.randint(0, 2, size=(10,)))
        pred = learner.predict(np.random.rand(10, 10))
        learner.predict_proba(np.random.rand(10, 10))
        confusion_matrix(pred, np.random.randint(0, 2, size=(10,)))


class TestBayesianOptimizer(unittest.TestCase):
    def test_set_max(self):
        # case 1: the estimator is not fitted yet
        regressor = mock.MockEstimator()
        learner = modAL.models.BayesianOptimizer(estimator=regressor)
        self.assertEqual(-np.inf, learner.max_val)

        # case 2: the estimator is fitted already
        for n_samples in range(1, 100):
            X = np.random.rand(n_samples, 2)
            y = np.random.rand(n_samples, )
            max_val = np.max(y)

            regressor = mock.MockEstimator()
            learner = modAL.models.BayesianOptimizer(
                estimator=regressor,
                X_training=X, y_training=y
            )
            np.testing.assert_almost_equal(max_val, learner.max_val)

    def test_set_new_max(self):
        for n_reps in range(100):
            # case 1: the learner is not fitted yet
            for n_samples in range(1, 10):
                y = np.random.rand(n_samples)
                regressor = mock.MockEstimator()
                learner = modAL.models.BayesianOptimizer(estimator=regressor)
                learner._set_max(y)
                self.assertEqual(learner.max_val, np.max(y))

            # case 2: new value is not a maximum
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 2)
                y = np.random.rand(n_samples)

                regressor = mock.MockEstimator()
                learner = modAL.models.BayesianOptimizer(
                    estimator=regressor,
                    X_training=X, y_training=y
                )

                y_new = y - np.random.rand()
                old_max = learner.max_val
                learner._set_max(y_new)
                np.testing.assert_almost_equal(old_max, learner.max_val)

            # case 3: new value is a maximum
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 2)
                y = np.random.rand(n_samples)

                regressor = mock.MockEstimator()
                learner = modAL.models.BayesianOptimizer(
                    estimator=regressor,
                    X_training=X, y_training=y
                )

                y_new = y + np.random.rand()
                learner._set_max(y_new)
                np.testing.assert_almost_equal(np.max(y_new), learner.max_val)

    def test_teach(self):
        for bootstrap, only_new in product([True, False], [True, False]):
            # case 1. optimizer is uninitialized
            for n_samples in range(1, 100):
                for n_features in range(1, 100):
                    regressor = mock.MockEstimator()
                    learner = modAL.models.BayesianOptimizer(estimator=regressor)

                    X = np.random.rand(n_samples, 2)
                    y = np.random.rand(n_samples)
                    learner.teach(X, y, bootstrap=bootstrap, only_new=only_new)

            # case 2. optimizer is initialized
            for n_samples in range(1, 100):
                for n_features in range(1, 100):
                    X = np.random.rand(n_samples, 2)
                    y = np.random.rand(n_samples)

                    regressor = mock.MockEstimator()
                    learner = modAL.models.BayesianOptimizer(
                        estimator=regressor,
                        X_training=X, y_training=y
                    )
                    learner.teach(X, y, bootstrap=bootstrap, only_new=only_new)


class TestCommittee(unittest.TestCase):

    def test_set_classes(self):
        for n_classes in range(1, 10):
            learner_list = [modAL.models.ActiveLearner(estimator=mock.MockEstimator(classes_=np.asarray([idx])))
                            for idx in range(n_classes)]
            committee = modAL.models.Committee(learner_list=learner_list)
            np.testing.assert_equal(
                committee.classes_,
                np.unique(range(n_classes))
            )

    def test_predict(self):
        for n_learners in range(1, 10):
            for n_instances in range(1, 10):
                prediction = np.random.randint(10, size=(n_instances, n_learners))
                committee = modAL.models.Committee(
                    learner_list=[mock.MockActiveLearner(
                                      mock.MockEstimator(classes_=np.asarray([0])),
                                      predict_return=prediction[:, learner_idx]
                                  )
                                  for learner_idx in range(n_learners)]
                )
                np.testing.assert_equal(
                    committee.vote(np.random.rand(n_instances, 5)),
                    prediction
                )

    def test_predict_proba(self):
        for n_samples in range(1, 100):
            for n_learners in range(1, 10):
                for n_classes in range(1, 10):
                    vote_proba_output = np.random.rand(n_samples, n_learners, n_classes)
                    # assembling the mock learners
                    learner_list = [mock.MockActiveLearner(
                        predict_proba_return=vote_proba_output[:, learner_idx, :],
                        predictor=mock.MockEstimator(classes_=list(range(n_classes)))
                    ) for learner_idx in range(n_learners)]
                    committee = modAL.models.Committee(learner_list=learner_list)
                    np.testing.assert_almost_equal(
                        committee.predict_proba(np.random.rand(n_samples, 1)),
                        np.mean(vote_proba_output, axis=1)
                    )

    def test_vote(self):
        for n_members in range(1, 10):
            for n_instances in range(1, 100):
                vote_output = np.random.randint(0, 2, size=(n_instances, n_members))
                # assembling the Committee
                learner_list = [mock.MockActiveLearner(
                                    predict_return=vote_output[:, member_idx],
                                    predictor=mock.MockEstimator(classes_=[0])
                                )
                                for member_idx in range(n_members)]
                committee = modAL.models.Committee(learner_list=learner_list)
                np.testing.assert_array_almost_equal(
                    committee.vote(np.random.rand(n_instances).reshape(-1, 1)),
                    vote_output
                )

    def test_vote_proba(self):
        for n_samples in range(1, 100):
            for n_learners in range(1, 10):
                for n_classes in range(1, 10):
                    vote_proba_output = np.random.rand(n_samples, n_learners, n_classes)
                    # assembling the mock learners
                    learner_list = [mock.MockActiveLearner(
                        predict_proba_return=vote_proba_output[:, learner_idx, :],
                        predictor=mock.MockEstimator(classes_=list(range(n_classes)))
                    ) for learner_idx in range(n_learners)]
                    committee = modAL.models.Committee(learner_list=learner_list)
                    np.testing.assert_almost_equal(
                        committee.vote_proba(np.random.rand(n_samples, 1)),
                        vote_proba_output
                    )

    def test_teach(self):
        X_training = np.random.rand(10, 2)
        y_training = np.random.randint(0, 2, size=10)

        for bootstrap, only_new in product([True, False], [True, False]):
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 2)
                y = np.random.randint(0, 2, size=n_samples)

                learner_1 = modAL.models.ActiveLearner(
                    X_training=X_training, y_training=y_training,
                    estimator=mock.MockEstimator(classes_=[0, 1])
                )
                learner_2 = modAL.models.ActiveLearner(
                    X_training=X_training, y_training=y_training,
                    estimator=mock.MockEstimator(classes_=[0, 1])
                )

                committee = modAL.models.Committee(
                    learner_list=[learner_1, learner_2]
                )

                committee.teach(X, y, bootstrap=bootstrap, only_new=only_new)


class TestCommitteeRegressor(unittest.TestCase):

    def test_predict(self):
        for n_members in range(1, 10):
            for n_instances in range(1, 100):
                vote = np.random.rand(n_instances, n_members)
                # assembling the Committee
                learner_list = [mock.MockActiveLearner(predict_return=vote[:, member_idx])
                                for member_idx in range(n_members)]
                committee = modAL.models.CommitteeRegressor(learner_list=learner_list)
                np.testing.assert_array_almost_equal(
                    committee.predict(np.random.rand(n_instances).reshape(-1, 1), return_std=False),
                    np.mean(vote, axis=1)
                )
                np.testing.assert_array_almost_equal(
                    committee.predict(np.random.rand(n_instances).reshape(-1, 1), return_std=True),
                    (np.mean(vote, axis=1), np.std(vote, axis=1))
                )

    def test_vote(self):
        for n_members in range(1, 10):
            for n_instances in range(1, 100):
                vote_output = np.random.rand(n_instances, n_members)
                # assembling the Committee
                learner_list = [mock.MockActiveLearner(predict_return=vote_output[:, member_idx])
                                for member_idx in range(n_members)]
                committee = modAL.models.CommitteeRegressor(learner_list=learner_list)
                np.testing.assert_array_almost_equal(
                    committee.vote(np.random.rand(n_instances).reshape(-1, 1)),
                    vote_output
                )


class TestExamples(unittest.TestCase):

    def test_examples(self):
        import example_tests.active_regression
        import example_tests.bagging
        import example_tests.ensemble
        import example_tests.ensemble_regression
        import example_tests.pool_based_sampling
        import example_tests.query_by_committee
        import example_tests.shape_learning
        import example_tests.stream_based_sampling
        import example_tests.custom_query_strategies
        import example_tests.information_density


if __name__ == '__main__':
    unittest.main(verbosity=2)
