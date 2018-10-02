import random
import unittest
import numpy as np

import mock
import modAL.models.base
import modAL.models.learners
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
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy, norm
from scipy.special import ndtr
from scipy import sparse as sp


Test = namedtuple('Test', ['input', 'output'])


def random_array(shape, n_arrays):
    for _ in range(n_arrays):
        yield np.random.rand(*shape)


class TestUtils(unittest.TestCase):

    def test_check_class_labels(self):
        for n_labels in range(1, 10):
            for n_learners in range(1, 10):
                # 1. test fitted estimators
                labels = np.random.randint(10, size=n_labels)
                different_labels = np.random.randint(10, 20, size=np.random.randint(1, 10))
                learner_list_1 = [mock.MockEstimator(classes_=labels) for _ in range(n_learners)]
                learner_list_2 = [mock.MockEstimator(classes_=different_labels) for _ in range(np.random.randint(1, 5))]
                shuffled_learners = random.sample(learner_list_1 + learner_list_2, len(learner_list_1 + learner_list_2))
                self.assertTrue(modAL.utils.validation.check_class_labels(*learner_list_1))
                self.assertFalse(modAL.utils.validation.check_class_labels(*shuffled_learners))

                # 2. test unfitted estimators
                unfitted_learner_list = [mock.MockEstimator(classes_=labels) for _ in range(n_learners)]
                idx = np.random.randint(0, n_learners)
                unfitted_learner_list.insert(idx, mock.MockEstimator(fitted=False))
                self.assertRaises(NotFittedError, modAL.utils.validation.check_class_labels, *unfitted_learner_list)

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

        def dummy_function(X_in):
            return np.ones(shape=(len(X_in), 1))

        for n_samples in range(2, 10):
            for n_features in range(1, 10):
                for n_functions in range(2, 10):
                    functions = [dummy_function for _ in range(n_functions)]
                    linear_combination = modAL.utils.combination.make_linear_combination(*functions)

                    X_in = np.random.rand(n_samples, n_features)
                    if n_samples == 1:
                        true_result = float(n_functions)
                    else:
                        true_result = n_functions*np.ones(shape=(n_samples, 1))

                    try:
                        np.testing.assert_almost_equal(linear_combination(X_in), true_result)
                    except:
                        linear_combination(X_in)

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

                learner = modAL.models.learners.ActiveLearner(
                    estimator=mock.MockEstimator(predict_proba_return=proba)
                )

                query_1 = query_strategy(learner, X)
                query_2 = modAL.uncertainty.uncertainty_sampling(learner, X)

                np.testing.assert_equal(query_1[0], query_2[0])
                np.testing.assert_almost_equal(query_1[1], query_2[1])

    def test_data_vstack(self):
        for n_samples, n_features in product(range(1, 10), range(1, 10)):
            # numpy arrays
            a, b = np.random.rand(n_samples, n_features), np.random.rand(n_samples, n_features)
            np.testing.assert_almost_equal(
                modAL.utils.data.data_vstack((a, b)),
                np.concatenate((a, b))
            )

            # sparse matrices
            for format in ['lil', 'csc', 'csr']:
                a, b = sp.random(n_samples, n_features, format=format), sp.random(n_samples, n_features, format=format)
                self.assertEqual((modAL.utils.data.data_vstack((a, b)) != sp.vstack((a, b))).sum(), 0)

        # not supported formats
        self.assertRaises(TypeError, modAL.utils.data.data_vstack, (1, 1))


class TestAcquisitionFunctions(unittest.TestCase):
    def test_acquisition_functions(self):
        for n_samples in range(1, 100):
            mean, std = np.random.rand(100, 1), np.random.rand(100, 1)
            modAL.acquisition.PI(mean, std, 0, 0)
            modAL.acquisition.EI(mean, std, 0, 0)
            modAL.acquisition.UCB(mean, std, 0)

            mean, std = np.random.rand(100, ), np.random.rand(100, )
            modAL.acquisition.PI(mean, std, 0, 0)
            modAL.acquisition.EI(mean, std, 0, 0)
            modAL.acquisition.UCB(mean, std, 0)

    def test_optimizer_PI(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, 1)
            std = np.random.rand(n_samples, 1)
            tradeoff = np.random.rand()
            max_val = np.random.rand()

            # 1. fitted estimator
            mock_estimator = mock.MockEstimator(predict_return=(mean, std))
            optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_PI = ndtr((mean - max_val - tradeoff)/std)

            np.testing.assert_almost_equal(
                true_PI,
                modAL.acquisition.optimizer_PI(optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

            # 2. unfitted estimator
            mock_estimator = mock.MockEstimator(fitted=False)
            optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_PI = ndtr((np.zeros(shape=(len(mean), 1)) - max_val - tradeoff) / np.ones(shape=(len(mean), 1)))

            np.testing.assert_almost_equal(
                true_PI,
                modAL.acquisition.optimizer_PI(optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

    def test_optimizer_EI(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, 1)
            std = np.random.rand(n_samples, 1)
            tradeoff = np.random.rand()
            max_val = np.random.rand()

            # 1. fitted estimator
            mock_estimator = mock.MockEstimator(
                predict_return=(mean, std)
            )
            optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_EI = (mean - optimizer.y_max - tradeoff) * ndtr((mean - optimizer.y_max - tradeoff) / std) \
                      + std * norm.pdf((mean - optimizer.y_max - tradeoff) / std)

            np.testing.assert_almost_equal(
                true_EI,
                modAL.acquisition.optimizer_EI(optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

            # 2. unfitted estimator
            mock_estimator = mock.MockEstimator(fitted=False)
            optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_EI = (np.zeros(shape=(len(mean), 1)) - optimizer.y_max - tradeoff) * ndtr((np.zeros(shape=(len(mean), 1)) - optimizer.y_max - tradeoff) / np.ones(shape=(len(mean), 1))) \
                      + np.ones(shape=(len(mean), 1)) * norm.pdf((np.zeros(shape=(len(mean), 1)) - optimizer.y_max - tradeoff) / np.ones(shape=(len(mean), 1)))

            np.testing.assert_almost_equal(
                true_EI,
                modAL.acquisition.optimizer_EI(optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

    def test_optimizer_UCB(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, 1)
            std = np.random.rand(n_samples, 1)
            beta = np.random.rand()

            # 1. fitted estimator
            mock_estimator = mock.MockEstimator(
                predict_return=(mean, std)
            )
            optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
            true_UCB = mean + beta*std

            np.testing.assert_almost_equal(
                true_UCB,
                modAL.acquisition.optimizer_UCB(optimizer, np.random.rand(n_samples, 2), beta)
            )

            # 2. unfitted estimator
            mock_estimator = mock.MockEstimator(fitted=False)
            optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
            true_UCB = np.zeros(shape=(len(mean), 1)) + beta * np.ones(shape=(len(mean), 1))

            np.testing.assert_almost_equal(
                true_UCB,
                modAL.acquisition.optimizer_UCB(optimizer, np.random.rand(n_samples, 2), beta)
            )

    def test_selection(self):
        for n_samples in range(1, 100):
            for n_instances in range(1, n_samples):
                X = np.random.rand(n_samples, 3)
                mean = np.random.rand(n_samples, )
                std = np.random.rand(n_samples, )
                max_val = np.random.rand()

                mock_estimator = mock.MockEstimator(
                    predict_return=(mean, std)
                )

                optimizer = modAL.models.learners.BayesianOptimizer(estimator=mock_estimator)
                optimizer._set_max([0], [max_val])

                modAL.acquisition.max_PI(optimizer, X, tradeoff=np.random.rand(), n_instances=n_instances)
                modAL.acquisition.max_EI(optimizer, X, tradeoff=np.random.rand(), n_instances=n_instances)
                modAL.acquisition.max_UCB(optimizer, X, beta=np.random.rand(), n_instances=n_instances)


class TestUncertainties(unittest.TestCase):

    def test_classifier_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k, l)), (1 - p) * np.ones(shape=(k, )))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        for case in test_cases:
            # fitted estimator
            fitted_estimator = mock.MockEstimator(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_uncertainty(fitted_estimator, np.random.rand(10)),
                case.output
            )

            # not fitted estimator
            not_fitted_estimator = mock.MockEstimator(fitted=False)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_uncertainty(not_fitted_estimator, case.input),
                np.ones(shape=(len(case.output)))
            )

    def test_classifier_margin(self):
        test_cases_1 = (Test(p * np.ones(shape=(k, l)), np.zeros(shape=(k,)))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        test_cases_2 = (Test(p * np.tile(np.asarray(range(k))+1.0, l).reshape(l, k),
                             p * np.ones(shape=(l, ))*int(k!=1))
                        for k in range(1, 10) for l in range(1, 100) for p in np.linspace(0, 1, 11))
        for case in chain(test_cases_1, test_cases_2):
            # fitted estimator
            fitted_estimator = mock.MockEstimator(predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_margin(fitted_estimator, np.random.rand(10)),
                case.output
            )

            # not fitted estimator
            not_fitted_estimator = mock.MockEstimator(fitted=False)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_margin(not_fitted_estimator, case.input),
                np.zeros(shape=(len(case.output)))
            )

    def test_classifier_entropy(self):
        for n_samples in range(1, 100):
            for n_classes in range(1, 20):
                proba = np.zeros(shape=(n_samples, n_classes))
                for sample_idx in range(n_samples):
                    proba[sample_idx, np.random.choice(range(n_classes))] = 1.0

                # fitted estimator
                fitted_estimator = mock.MockEstimator(predict_proba_return=proba)
                np.testing.assert_equal(
                    modAL.uncertainty.classifier_entropy(fitted_estimator, np.random.rand(n_samples, 1)),
                    np.zeros(shape=(n_samples, ))
                )

                # not fitted estimator
                not_fitted_estimator = mock.MockEstimator(fitted=False)
                np.testing.assert_almost_equal(
                    modAL.uncertainty.classifier_entropy(not_fitted_estimator, np.random.rand(n_samples, 1)),
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
            for n_classes in range(2, 10):
                for true_query_idx in range(n_samples):
                    predict_proba = np.zeros(shape=(n_samples, n_classes))
                    predict_proba[:, 0] = 1.0
                    predict_proba[true_query_idx, 0] = 0.0
                    classifier = mock.MockEstimator(predict_proba_return=predict_proba)
                    query_idx, query_instance = modAL.uncertainty.margin_sampling(
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
                    query_idx, query_instance = modAL.uncertainty.entropy_sampling(
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
                    # 1. fitted committee
                    vote_return = np.zeros(shape=(n_samples, n_classes), dtype=np.int16)
                    vote_return[true_query_idx] = np.asarray(range(n_classes), dtype=np.int16)
                    committee = mock.MockCommittee(classes_=np.asarray(range(n_classes)), vote_return=vote_return)
                    vote_entr = modAL.disagreement.vote_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    true_entropy = np.zeros(shape=(n_samples, ))
                    true_entropy[true_query_idx] = entropy(np.ones(n_classes)/n_classes)
                    np.testing.assert_array_almost_equal(vote_entr, true_entropy)

                    # 2. unfitted committee
                    committee = mock.MockCommittee(fitted=False)
                    true_entropy = np.zeros(shape=(n_samples,))
                    vote_entr = modAL.disagreement.vote_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_almost_equal(vote_entr, true_entropy)

    def test_consensus_entropy(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                for true_query_idx in range(n_samples):
                    # 1. fitted committee
                    proba = np.zeros(shape=(n_samples, n_classes))
                    proba[:, 0] = 1.0
                    proba[true_query_idx] = np.ones(n_classes)/n_classes
                    committee = mock.MockCommittee(predict_proba_return=proba)
                    consensus_entropy = modAL.disagreement.consensus_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    true_entropy = np.zeros(shape=(n_samples,))
                    true_entropy[true_query_idx] = entropy(np.ones(n_classes) / n_classes)
                    np.testing.assert_array_almost_equal(consensus_entropy, true_entropy)

                    # 2. unfitted committee
                    committee = mock.MockCommittee(fitted=False)
                    true_entropy = np.zeros(shape=(n_samples,))
                    consensus_entropy = modAL.disagreement.consensus_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_almost_equal(consensus_entropy, true_entropy)

    def test_KL_max_disagreement(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                for n_learners in range (2, 10):
                    # 1. fitted committee
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

                    # 2. unfitted committee
                    committee = mock.MockCommittee(fitted=False)
                    true_KL_disagreement = np.zeros(shape=(n_samples,))
                    returned_KL_disagreement = modAL.disagreement.KL_max_disagreement(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_almost_equal(returned_KL_disagreement, true_KL_disagreement)


class TestQueries(unittest.TestCase):

    def test_multi_argmax(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool):
                utility = np.zeros(n_pool)
                max_idx = np.random.choice(range(n_pool), size=n_instances, replace=False)
                utility[max_idx] = 1e-10 + np.random.rand(n_instances, )
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
                    learner = modAL.models.learners.ActiveLearner(
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
                    learner = modAL.models.learners.ActiveLearner(
                        estimator=mock.MockEstimator(),
                        X_training=X_initial, y_training=y_initial
                    )
                    learner._add_training_data(X_new, y_new)
                    np.testing.assert_equal(
                        learner.y_training,
                        np.concatenate((y_initial, y_new))
                    )
                    # 3. data with shape (n, )
                    X_initial = np.random.rand(n_samples, )
                    y_initial = np.random.randint(0, 2, size=(n_samples,))
                    learner = modAL.models.learners.ActiveLearner(
                        estimator=mock.MockEstimator(),
                        X_training=X_initial, y_training=y_initial
                    )
                    X_new = np.random.rand(n_new_samples,)
                    y_new = np.random.randint(0, 2, size=(n_new_samples,))
                    learner._add_training_data(X_new, y_new)



                    # testing for invalid cases
                    # 1. len(X_new) != len(y_new)
                    X_new = np.random.rand(n_new_samples, n_features)
                    y_new = np.random.randint(0, 2, size=(2*n_new_samples,))
                    self.assertRaises(ValueError, learner._add_training_data, X_new, y_new)
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
                learner = modAL.models.learners.ActiveLearner(
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
                learner = modAL.models.learners.ActiveLearner(
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
                learner = modAL.models.learners.ActiveLearner(
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
            learner = modAL.models.learners.ActiveLearner(mock_classifier, mock.MockFunction(None))
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

                learner = modAL.models.learners.ActiveLearner(
                    X_training=X_training, y_training=y_training,
                    estimator=mock.MockEstimator()
                )

                learner.teach(X, y, bootstrap=bootstrap, only_new=only_new)

    def test_keras(self):
        pass

    def test_sklearn(self):
        learner = modAL.models.learners.ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=np.random.rand(10, 10),
            y_training=np.random.randint(0, 2, size=(10,))
        )
        learner.fit(np.random.rand(10, 10), np.random.randint(0, 2, size=(10,)))
        pred = learner.predict(np.random.rand(10, 10))
        learner.predict_proba(np.random.rand(10, 10))
        confusion_matrix(pred, np.random.randint(0, 2, size=(10,)))

    def test_sparse_matrices(self):
        query_strategies = [
            modAL.uncertainty.uncertainty_sampling,
            modAL.uncertainty.entropy_sampling,
            modAL.uncertainty.margin_sampling
        ]
        formats = ['lil', 'csc', 'csr']
        sample_count = range(10, 20)
        feature_count = range(1, 5)

        for query_strategy, format, n_samples, n_features in product(query_strategies, formats, sample_count, feature_count):
            X_pool = sp.random(n_samples, n_features, format=format)
            y_pool = np.random.randint(0, 2, size=(n_samples, ))
            initial_idx = np.random.choice(range(n_samples), size=5, replace=False)

            learner = modAL.models.learners.ActiveLearner(
                estimator=RandomForestClassifier(), query_strategy=query_strategy,
                X_training=X_pool[initial_idx], y_training=y_pool[initial_idx]
            )
            query_idx, query_inst = learner.query(X_pool)
            learner.teach(X_pool[query_idx], y_pool[query_idx])


class TestBayesianOptimizer(unittest.TestCase):
    def test_set_max(self):
        # case 1: the estimator is not fitted yet
        regressor = mock.MockEstimator()
        learner = modAL.models.learners.BayesianOptimizer(estimator=regressor)
        self.assertEqual(-np.inf, learner.y_max)

        # case 2: the estimator is fitted already
        for n_samples in range(1, 100):
            X = np.random.rand(n_samples, 2)
            y = np.random.rand(n_samples, )
            max_val = np.max(y)

            regressor = mock.MockEstimator()
            learner = modAL.models.learners.BayesianOptimizer(
                estimator=regressor,
                X_training=X, y_training=y
            )
            np.testing.assert_almost_equal(max_val, learner.y_max)

    def test_set_new_max(self):
        for n_reps in range(100):
            # case 1: the learner is not fitted yet
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 3)
                y = np.random.rand(n_samples)
                max_idx = np.argmax(y)
                regressor = mock.MockEstimator()
                learner = modAL.models.learners.BayesianOptimizer(estimator=regressor)
                learner._set_max(X, y)
                np.testing.assert_equal(learner.X_max, X[max_idx])
                np.testing.assert_equal(learner.y_max, y[max_idx])

            # case 2: new value is not a maximum
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 2)
                y = np.random.rand(n_samples)

                regressor = mock.MockEstimator()
                learner = modAL.models.learners.BayesianOptimizer(
                    estimator=regressor,
                    X_training=X, y_training=y
                )

                X_new = np.random.rand()
                y_new = y - np.random.rand()
                X_old_max = learner.X_max
                y_old_max = learner.y_max
                learner._set_max(X_new, y_new)
                np.testing.assert_equal(X_old_max, learner.X_max)
                np.testing.assert_equal(y_old_max, learner.y_max)

            # case 3: new value is a maximum
            for n_samples in range(1, 10):
                X = np.random.rand(n_samples, 2)
                y = np.random.rand(n_samples)

                regressor = mock.MockEstimator()
                learner = modAL.models.learners.BayesianOptimizer(
                    estimator=regressor,
                    X_training=X, y_training=y
                )

                X_new = np.random.rand(n_samples, 2)
                y_new = y + np.random.rand()
                max_idx = np.argmax(y_new)
                learner._set_max(X_new, y_new)
                np.testing.assert_equal(X_new[max_idx], learner.X_max)
                np.testing.assert_equal(y_new[max_idx], learner.y_max)

    def test_get_max(self):
        for n_samples in range(1, 100):
            for max_idx in range(0, n_samples):
                X = np.random.rand(n_samples, 3)
                y = np.random.rand(n_samples)
                y[max_idx] = 10

                regressor = mock.MockEstimator()
                optimizer = modAL.models.learners.BayesianOptimizer(regressor, X_training=X, y_training=y)
                X_max, y_max = optimizer.get_max()
                np.testing.assert_equal(X_max, X[max_idx])
                np.testing.assert_equal(y_max, y[max_idx])

    def test_teach(self):
        for bootstrap, only_new in product([True, False], [True, False]):
            # case 1. optimizer is uninitialized
            for n_samples in range(1, 100):
                for n_features in range(1, 100):
                    regressor = mock.MockEstimator()
                    learner = modAL.models.learners.BayesianOptimizer(estimator=regressor)

                    X = np.random.rand(n_samples, 2)
                    y = np.random.rand(n_samples)
                    learner.teach(X, y, bootstrap=bootstrap, only_new=only_new)

            # case 2. optimizer is initialized
            for n_samples in range(1, 100):
                for n_features in range(1, 100):
                    X = np.random.rand(n_samples, 2)
                    y = np.random.rand(n_samples)

                    regressor = mock.MockEstimator()
                    learner = modAL.models.learners.BayesianOptimizer(
                        estimator=regressor,
                        X_training=X, y_training=y
                    )
                    learner.teach(X, y, bootstrap=bootstrap, only_new=only_new)


class TestCommittee(unittest.TestCase):

    def test_set_classes(self):
        # 1. test unfitted learners
        for n_learners in range(1, 10):
            learner_list = [modAL.models.learners.ActiveLearner(estimator=mock.MockEstimator(fitted=False))
                            for idx in range(n_learners)]
            committee = modAL.models.learners.Committee(learner_list=learner_list)
            self.assertEqual(committee.classes_, None)
            self.assertEqual(committee.n_classes_, 0)

        # 2. test fitted learners
        for n_classes in range(1, 10):
            learner_list = [modAL.models.learners.ActiveLearner(estimator=mock.MockEstimator(classes_=np.asarray([idx])))
                            for idx in range(n_classes)]
            committee = modAL.models.learners.Committee(learner_list=learner_list)
            np.testing.assert_equal(
                committee.classes_,
                np.unique(range(n_classes))
            )

    def test_predict(self):
        for n_learners in range(1, 10):
            for n_instances in range(1, 10):
                prediction = np.random.randint(10, size=(n_instances, n_learners))
                committee = modAL.models.learners.Committee(
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
                    committee = modAL.models.learners.Committee(learner_list=learner_list)
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
                committee = modAL.models.learners.Committee(learner_list=learner_list)
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
                    committee = modAL.models.learners.Committee(learner_list=learner_list)
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

                learner_1 = modAL.models.learners.ActiveLearner(
                    X_training=X_training, y_training=y_training,
                    estimator=mock.MockEstimator(classes_=[0, 1])
                )
                learner_2 = modAL.models.learners.ActiveLearner(
                    X_training=X_training, y_training=y_training,
                    estimator=mock.MockEstimator(classes_=[0, 1])
                )

                committee = modAL.models.learners.Committee(
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
                committee = modAL.models.learners.CommitteeRegressor(learner_list=learner_list)
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
                committee = modAL.models.learners.CommitteeRegressor(learner_list=learner_list)
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
        import example_tests.bayesian_optimization
        import example_tests.ranked_batch_mode


if __name__ == '__main__':
    unittest.main(verbosity=2)
