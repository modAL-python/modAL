import random
import unittest
from collections import namedtuple
from copy import deepcopy
from itertools import chain, product
from unittest.mock import MagicMock

import modAL.acquisition
import modAL.batch
import modAL.density
import modAL.disagreement
import modAL.dropout
import modAL.expected_error
import modAL.models.base
import modAL.models.learners
import modAL.multilabel
import modAL.uncertainty
import modAL.utils.combination
import modAL.utils.selection
import modAL.utils.validation
import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from scipy.special import ndtr
from scipy.stats import entropy, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from torch import nn

import mock

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
                different_labels = np.random.randint(
                    10, 20, size=np.random.randint(1, 10))
                learner_list_1 = [mock.MockEstimator(
                    classes_=labels) for _ in range(n_learners)]
                learner_list_2 = [mock.MockEstimator(
                    classes_=different_labels) for _ in range(np.random.randint(1, 5))]
                shuffled_learners = random.sample(
                    learner_list_1 + learner_list_2, len(learner_list_1 + learner_list_2))
                self.assertTrue(
                    modAL.utils.validation.check_class_labels(*learner_list_1))
                self.assertFalse(
                    modAL.utils.validation.check_class_labels(*shuffled_learners))

                # 2. test unfitted estimators
                unfitted_learner_list = [mock.MockEstimator(
                    classes_=labels) for _ in range(n_learners)]
                idx = np.random.randint(0, n_learners)
                unfitted_learner_list.insert(
                    idx, mock.MockEstimator(fitted=False))
                self.assertRaises(
                    NotFittedError, modAL.utils.validation.check_class_labels, *unfitted_learner_list)

    def test_check_class_proba(self):
        for n_labels in range(2, 20):
            # when all classes are known:
            proba = np.random.rand(100, n_labels)
            class_labels = list(range(n_labels))
            np.testing.assert_almost_equal(
                modAL.utils.check_class_proba(
                    proba, known_labels=class_labels, all_labels=class_labels),
                proba
            )
            for unknown_idx in range(n_labels):
                all_labels = list(range(n_labels))
                known_labels = deepcopy(all_labels)
                known_labels.remove(unknown_idx)
                aug_proba = np.insert(
                    proba[:, known_labels], unknown_idx, np.zeros(len(proba)), axis=1)
                np.testing.assert_almost_equal(
                    modAL.utils.check_class_proba(
                        proba[:, known_labels], known_labels=known_labels, all_labels=all_labels),
                    aug_proba
                )

    def test_linear_combination(self):

        def dummy_function(X_in):
            return np.ones(shape=(len(X_in), 1))

        for n_samples in range(2, 10):
            for n_features in range(1, 10):
                for n_functions in range(2, 10):
                    functions = [dummy_function for _ in range(n_functions)]
                    linear_combination = modAL.utils.combination.make_linear_combination(
                        *functions)

                    X_in = np.random.rand(n_samples, n_features)
                    if n_samples == 1:
                        true_result = float(n_functions)
                    else:
                        true_result = n_functions*np.ones(shape=(n_samples, 1))

                    np.testing.assert_almost_equal(
                        linear_combination(X_in), true_result)

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
                exp_product = modAL.utils.combination.make_product(
                    *functions, exponents=exponents)
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

                np.testing.assert_equal(query_1, query_2)

    def test_data_vstack(self):
        for n_samples, n_features in product(range(1, 10), range(1, 10)):
            # numpy arrays
            a, b = np.random.rand(n_samples, n_features), np.random.rand(
                n_samples, n_features)
            np.testing.assert_almost_equal(
                modAL.utils.data.data_vstack((a, b)),
                np.concatenate((a, b))
            )

            # sparse matrices
            for format in ['lil', 'csc', 'csr']:
                a, b = sp.random(n_samples, n_features, format=format), sp.random(
                    n_samples, n_features, format=format)
                self.assertEqual((modAL.utils.data.data_vstack(
                    (a, b)) != sp.vstack((a, b))).sum(), 0)

            # lists
            a, b = np.random.rand(n_samples, n_features).tolist(), np.random.rand(
                n_samples, n_features).tolist()
            np.testing.assert_almost_equal(
                modAL.utils.data.data_vstack((a, b)),
                np.concatenate((a, b))
            )

            # torch.Tensors
            a, b = torch.ones(2, 2), torch.ones(2, 2)
            torch.testing.assert_allclose(
                modAL.utils.data.data_vstack((a, b)),
                torch.cat((a, b))
            )

        # not supported formats
        self.assertRaises(TypeError, modAL.utils.data.data_vstack, (1, 1))

    # functions from modALu.tils.selection

    def test_multi_argmax(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool+1):
                utility = np.zeros(n_pool)
                max_idx = np.random.choice(
                    range(n_pool), size=n_instances, replace=False)
                utility[max_idx] = 1e-10 + np.random.rand(n_instances, )
                np.testing.assert_equal(
                    np.sort(modAL.utils.selection.multi_argmax(
                        utility, n_instances)),
                    (np.sort(max_idx), np.sort(utility)
                     [len(utility)-n_instances:])
                )

    def test_shuffled_argmax(self):
        for n_pool in range(1, 100):
            for n_instances in range(1, n_pool+1):
                values = np.random.permutation(n_pool)
                true_query_idx = np.argsort(values)[len(values)-n_instances:]
                true_values = np.sort(values, axis=None)[
                    len(values)-n_instances:]

                np.testing.assert_equal(
                    (true_query_idx, true_values),
                    modAL.utils.selection.shuffled_argmax(values, n_instances)
                )

    def test_weighted_random(self):
        for n_pool in range(2, 100):
            for n_instances in range(1, n_pool):
                utility = np.ones(n_pool)
                query_idx = modAL.utils.selection.weighted_random(
                    utility, n_instances)
                # testing for correct number of returned indices
                np.testing.assert_equal(len(query_idx), n_instances)
                # testing for uniqueness of each query index
                np.testing.assert_equal(
                    len(query_idx), len(np.unique(query_idx)))


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
            mean = np.random.rand(n_samples, )
            std = np.random.rand(n_samples, )
            tradeoff = np.random.rand()
            max_val = np.random.rand()

            # 1. fitted estimator
            mock_estimator = mock.MockEstimator(predict_return=(mean, std))
            optimizer = modAL.models.learners.BayesianOptimizer(
                estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_PI = ndtr((mean - max_val - tradeoff)/std)

            np.testing.assert_almost_equal(
                true_PI,
                modAL.acquisition.optimizer_PI(
                    optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

            # 2. unfitted estimator
            mock_estimator = mock.MockEstimator(fitted=False)
            optimizer = modAL.models.learners.BayesianOptimizer(
                estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_PI = ndtr((np.zeros(shape=(len(mean), 1)) -
                           max_val - tradeoff) / np.ones(shape=(len(mean), 1)))

            np.testing.assert_almost_equal(
                true_PI,
                modAL.acquisition.optimizer_PI(
                    optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

    def test_optimizer_EI(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, )
            std = np.random.rand(n_samples, )
            tradeoff = np.random.rand()
            max_val = np.random.rand()

            # 1. fitted estimator
            mock_estimator = mock.MockEstimator(
                predict_return=(mean, std)
            )
            optimizer = modAL.models.learners.BayesianOptimizer(
                estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_EI = (mean - optimizer.y_max - tradeoff) * ndtr((mean - optimizer.y_max - tradeoff) / std) \
                + std * norm.pdf((mean - optimizer.y_max - tradeoff) / std)

            np.testing.assert_almost_equal(
                true_EI,
                modAL.acquisition.optimizer_EI(
                    optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

            # 2. unfitted estimator
            mock_estimator = mock.MockEstimator(fitted=False)
            optimizer = modAL.models.learners.BayesianOptimizer(
                estimator=mock_estimator)
            optimizer._set_max([0], [max_val])
            true_EI = (np.zeros(shape=(len(mean), 1)) - optimizer.y_max - tradeoff) * ndtr((np.zeros(shape=(len(mean), 1)) - optimizer.y_max - tradeoff) / np.ones(shape=(len(mean), 1))) \
                + np.ones(shape=(len(mean), 1)) * norm.pdf((np.zeros(shape=(len(mean), 1)
                                                                     ) - optimizer.y_max - tradeoff) / np.ones(shape=(len(mean), 1)))

            np.testing.assert_almost_equal(
                true_EI,
                modAL.acquisition.optimizer_EI(
                    optimizer, np.random.rand(n_samples, 2), tradeoff)
            )

    def test_optimizer_UCB(self):
        for n_samples in range(1, 100):
            mean = np.random.rand(n_samples, )
            std = np.random.rand(n_samples, )
            beta = np.random.rand()

            # 1. fitted estimator
            mock_estimator = mock.MockEstimator(
                predict_return=(mean, std)
            )
            optimizer = modAL.models.learners.BayesianOptimizer(
                estimator=mock_estimator)
            true_UCB = mean + beta*std

            np.testing.assert_almost_equal(
                true_UCB,
                modAL.acquisition.optimizer_UCB(
                    optimizer, np.random.rand(n_samples, 2), beta)
            )

            # 2. unfitted estimator
            mock_estimator = mock.MockEstimator(fitted=False)
            optimizer = modAL.models.learners.BayesianOptimizer(
                estimator=mock_estimator)
            true_UCB = np.zeros(shape=(len(mean), 1)) + \
                beta * np.ones(shape=(len(mean), 1))

            np.testing.assert_almost_equal(
                true_UCB,
                modAL.acquisition.optimizer_UCB(
                    optimizer, np.random.rand(n_samples, 2), beta)
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

                optimizer = modAL.models.learners.BayesianOptimizer(
                    estimator=mock_estimator)
                optimizer._set_max([0], [max_val])

                modAL.acquisition.max_PI(
                    optimizer, X, tradeoff=np.random.rand(), n_instances=n_instances)
                modAL.acquisition.max_EI(
                    optimizer, X, tradeoff=np.random.rand(), n_instances=n_instances)
                modAL.acquisition.max_UCB(
                    optimizer, X, beta=np.random.rand(), n_instances=n_instances)


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
                    vote_return = np.zeros(
                        shape=(n_samples, n_classes), dtype=np.int16)
                    vote_return[true_query_idx] = np.asarray(
                        range(n_classes), dtype=np.int16)
                    committee = mock.MockCommittee(classes_=np.asarray(
                        range(n_classes)), vote_return=vote_return)
                    vote_entr = modAL.disagreement.vote_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    true_entropy = np.zeros(shape=(n_samples, ))
                    true_entropy[true_query_idx] = entropy(
                        np.ones(n_classes)/n_classes)
                    np.testing.assert_array_almost_equal(
                        vote_entr, true_entropy)

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
                    true_entropy[true_query_idx] = entropy(
                        np.ones(n_classes) / n_classes)
                    np.testing.assert_array_almost_equal(
                        consensus_entropy, true_entropy)

                    # 2. unfitted committee
                    committee = mock.MockCommittee(fitted=False)
                    true_entropy = np.zeros(shape=(n_samples,))
                    consensus_entropy = modAL.disagreement.consensus_entropy(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_almost_equal(
                        consensus_entropy, true_entropy)

    def test_KL_max_disagreement(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                for n_learners in range(2, 10):
                    # 1. fitted committee
                    vote_proba = np.zeros(
                        shape=(n_samples, n_learners, n_classes))
                    vote_proba[:, :, 0] = 1.0
                    committee = mock.MockCommittee(
                        n_learners=n_learners, classes_=range(n_classes),
                        vote_proba_return=vote_proba
                    )

                    true_KL_disagreement = np.zeros(shape=(n_samples, ))

                    np.testing.assert_array_almost_equal(
                        true_KL_disagreement,
                        modAL.disagreement.KL_max_disagreement(
                            committee, np.random.rand(n_samples, 1))
                    )

                    # 2. unfitted committee
                    committee = mock.MockCommittee(fitted=False)
                    true_KL_disagreement = np.zeros(shape=(n_samples,))
                    returned_KL_disagreement = modAL.disagreement.KL_max_disagreement(
                        committee, np.random.rand(n_samples, n_classes)
                    )
                    np.testing.assert_almost_equal(
                        returned_KL_disagreement, true_KL_disagreement)

    def test_vote_entropy_sampling(self):
        for n_samples, n_features, n_classes in product(range(1, 10), range(1, 10), range(1, 10)):
            committee = mock.MockCommittee(classes_=np.asarray(range(n_classes)),
                                           vote_return=np.zeros(shape=(n_samples, n_classes), dtype=np.int16))
            modAL.disagreement.vote_entropy_sampling(
                committee, np.random.rand(n_samples, n_features))
            modAL.disagreement.vote_entropy_sampling(committee, np.random.rand(n_samples, n_features),
                                                     random_tie_break=True)

    def test_consensus_entropy_sampling(self):
        for n_samples, n_features, n_classes in product(range(1, 10), range(1, 10), range(1, 10)):
            committee = mock.MockCommittee(
                predict_proba_return=np.random.rand(n_samples, n_classes))
            modAL.disagreement.consensus_entropy_sampling(
                committee, np.random.rand(n_samples, n_features))
            modAL.disagreement.consensus_entropy_sampling(committee, np.random.rand(n_samples, n_features),
                                                          random_tie_break=True)

    def test_max_disagreement_sampling(self):
        for n_samples, n_features, n_classes, n_learners in product(range(1, 10), range(1, 10), range(1, 10), range(2, 5)):
            committee = mock.MockCommittee(
                n_learners=n_learners, classes_=range(n_classes),
                vote_proba_return=np.zeros(
                    shape=(n_samples, n_learners, n_classes))
            )
            modAL.disagreement.max_disagreement_sampling(
                committee, np.random.rand(n_samples, n_features))
            modAL.disagreement.max_disagreement_sampling(committee, np.random.rand(n_samples, n_features),
                                                         random_tie_break=True)

    def test_max_std_sampling(self):
        for n_samples, n_features in product(range(1, 10), range(1, 10)):
            regressor = GaussianProcessRegressor()
            regressor.fit(np.random.rand(n_samples, n_features),
                          np.random.rand(n_samples))
            modAL.disagreement.max_std_sampling(
                regressor, np.random.rand(n_samples, n_features))
            modAL.disagreement.max_std_sampling(regressor, np.random.rand(n_samples, n_features),
                                                random_tie_break=True)


class TestEER(unittest.TestCase):
    def test_eer(self):
        for n_pool, n_features, n_classes in product(range(5, 10), range(1, 5), range(2, 5)):
            X_training_, y_training = np.random.rand(
                10, n_features).tolist(), np.random.randint(0, n_classes, size=10)
            X_pool_, y_pool = np.random.rand(n_pool, n_features).tolist(
            ), np.random.randint(0, n_classes+1, size=n_pool)

            for data_type in (sp.csr_matrix, pd.DataFrame, np.array, list):
                X_training, X_pool = data_type(X_training_), data_type(X_pool_)

                learner = modAL.models.ActiveLearner(RandomForestClassifier(n_estimators=2),
                                                     X_training=X_training, y_training=y_training)

                modAL.expected_error.expected_error_reduction(learner, X_pool)
                modAL.expected_error.expected_error_reduction(
                    learner, X_pool, random_tie_break=True)
                modAL.expected_error.expected_error_reduction(
                    learner, X_pool, p_subsample=0.1)
                modAL.expected_error.expected_error_reduction(
                    learner, X_pool, loss='binary')
                modAL.expected_error.expected_error_reduction(
                    learner, X_pool, p_subsample=0.1, loss='log')
                self.assertRaises(AssertionError, modAL.expected_error.expected_error_reduction,
                                  learner, X_pool, p_subsample=1.5)
                self.assertRaises(AssertionError, modAL.expected_error.expected_error_reduction,
                                  learner, X_pool, loss=42)


class TestUncertainties(unittest.TestCase):

    def test_classifier_uncertainty(self):
        test_cases = (Test(p * np.ones(shape=(k, l)), (1 - p) * np.ones(shape=(k, )))
                      for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        for case in test_cases:
            # testing _proba_uncertainty
            np.testing.assert_almost_equal(
                modAL.uncertainty._proba_uncertainty(case.input),
                case.output
            )

            # fitted estimator
            fitted_estimator = mock.MockEstimator(
                predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_uncertainty(
                    fitted_estimator, np.random.rand(10)),
                case.output
            )

            # not fitted estimator
            not_fitted_estimator = mock.MockEstimator(fitted=False)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_uncertainty(
                    not_fitted_estimator, case.input),
                np.ones(shape=(len(case.output)))
            )

    def test_classifier_margin(self):
        test_cases_1 = (Test(p * np.ones(shape=(k, l)), np.zeros(shape=(k,)))
                        for k in range(1, 100) for l in range(1, 10) for p in np.linspace(0, 1, 11))
        test_cases_2 = (Test(p * np.tile(np.asarray(range(k))+1.0, l).reshape(l, k),
                             p * np.ones(shape=(l, ))*int(k != 1))
                        for k in range(1, 10) for l in range(1, 100) for p in np.linspace(0, 1, 11))
        for case in chain(test_cases_1, test_cases_2):
            # _proba_margin
            np.testing.assert_almost_equal(
                modAL.uncertainty._proba_margin(case.input),
                case.output
            )

            # fitted estimator
            fitted_estimator = mock.MockEstimator(
                predict_proba_return=case.input)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_margin(
                    fitted_estimator, np.random.rand(10)),
                case.output
            )

            # not fitted estimator
            not_fitted_estimator = mock.MockEstimator(fitted=False)
            np.testing.assert_almost_equal(
                modAL.uncertainty.classifier_margin(
                    not_fitted_estimator, case.input),
                np.zeros(shape=(len(case.output)))
            )

    def test_classifier_entropy(self):
        for n_samples in range(1, 100):
            for n_classes in range(1, 20):
                proba = np.zeros(shape=(n_samples, n_classes))
                for sample_idx in range(n_samples):
                    proba[sample_idx, np.random.choice(range(n_classes))] = 1.0

                # _proba_entropy
                np.testing.assert_almost_equal(
                    modAL.uncertainty._proba_entropy(proba),
                    np.zeros(shape=(n_samples,))
                )

                # fitted estimator
                fitted_estimator = mock.MockEstimator(
                    predict_proba_return=proba)
                np.testing.assert_equal(
                    modAL.uncertainty.classifier_entropy(
                        fitted_estimator, np.random.rand(n_samples, 1)),
                    np.zeros(shape=(n_samples, ))
                )

                # not fitted estimator
                not_fitted_estimator = mock.MockEstimator(fitted=False)
                np.testing.assert_almost_equal(
                    modAL.uncertainty.classifier_entropy(
                        not_fitted_estimator, np.random.rand(n_samples, 1)),
                    np.zeros(shape=(n_samples, ))
                )

    def test_uncertainty_sampling(self):
        for n_samples in range(1, 10):
            for n_classes in range(1, 10):
                max_proba = np.zeros(n_classes)
                for true_query_idx in range(n_samples):
                    predict_proba = np.random.rand(n_samples, n_classes)
                    predict_proba[true_query_idx] = max_proba
                    classifier = mock.MockEstimator(
                        predict_proba_return=predict_proba)
                    query_idx, query_metric = modAL.uncertainty.uncertainty_sampling(
                        classifier, np.random.rand(n_samples, n_classes)
                    )
                    shuffled_query_idx, shuffled_query_metric = modAL.uncertainty.uncertainty_sampling(
                        classifier, np.random.rand(n_samples, n_classes),
                        random_tie_break=True
                    )
                    np.testing.assert_array_equal(query_idx, true_query_idx)
                    np.testing.assert_array_equal(
                        shuffled_query_idx, true_query_idx)

    def test_margin_sampling(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                for true_query_idx in range(n_samples):
                    predict_proba = np.zeros(shape=(n_samples, n_classes))
                    predict_proba[:, 0] = 1.0
                    predict_proba[true_query_idx, 0] = 0.0
                    classifier = mock.MockEstimator(
                        predict_proba_return=predict_proba)

                    query_idx, query_metric = modAL.uncertainty.margin_sampling(
                        classifier, np.random.rand(n_samples, n_classes)
                    )
                    shuffled_query_idx, shuffled_query_metric = modAL.uncertainty.margin_sampling(
                        classifier, np.random.rand(n_samples, n_classes),
                        random_tie_break=True
                    )
                    np.testing.assert_array_equal(query_idx, true_query_idx)
                    np.testing.assert_array_equal(
                        shuffled_query_idx, true_query_idx)

    def test_entropy_sampling(self):
        for n_samples in range(1, 10):
            for n_classes in range(2, 10):
                max_proba = np.ones(n_classes)/n_classes
                for true_query_idx in range(n_samples):
                    predict_proba = np.zeros(shape=(n_samples, n_classes))
                    predict_proba[:, 0] = 1.0
                    predict_proba[true_query_idx] = max_proba
                    classifier = mock.MockEstimator(
                        predict_proba_return=predict_proba)

                    query_idx, query_metric = modAL.uncertainty.entropy_sampling(
                        classifier, np.random.rand(n_samples, n_classes)
                    )
                    shuffled_query_idx, shuffled_query_metric = modAL.uncertainty.entropy_sampling(
                        classifier, np.random.rand(n_samples, n_classes),
                        random_tie_break=True
                    )
                    np.testing.assert_array_equal(query_idx, true_query_idx)
                    np.testing.assert_array_equal(
                        shuffled_query_idx, true_query_idx)


# PyTorch model for test cases --> Do not change the layers
class Torch_Model(nn.Module):
    def __init__(self,):
        super(Torch_Model, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fcs = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return x


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.skorch_classifier = NeuralNetClassifier(Torch_Model,
                                                     criterion=torch.nn.CrossEntropyLoss,
                                                     optimizer=torch.optim.Adam,
                                                     train_split=None,
                                                     verbose=1)

    def test_mc_dropout_bald(self):
        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.skorch_classifier,
            query_strategy=modAL.dropout.mc_dropout_bald,
        )
        for random_tie_break in [True, False]:
            for num_cycles, sample_per_forward_pass in product(range(1, 5), range(1, 5)):
                for n_samples, n_classes in product(range(1, 5), range(1, 5)):
                    for n_instances in range(1, n_samples):
                        X_pool = torch.randn(n_samples, n_classes)
                        modAL.dropout.mc_dropout_bald(learner, X_pool, n_instances, random_tie_break, [],
                                                      num_cycles, sample_per_forward_pass)

    def test_mc_dropout_mean_st(self):
        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.skorch_classifier,
            query_strategy=modAL.dropout.mc_dropout_mean_st,
        )
        for random_tie_break in [True, False]:
            for num_cycles, sample_per_forward_pass in product(range(1, 5), range(1, 5)):
                for n_samples, n_classes in product(range(1, 5), range(1, 5)):
                    for n_instances in range(1, n_samples):
                        X_pool = torch.randn(n_samples, n_classes)
                        modAL.dropout.mc_dropout_mean_st(learner, X_pool, n_instances, random_tie_break, [],
                                                         num_cycles, sample_per_forward_pass)

    def test_mc_dropout_max_entropy(self):
        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.skorch_classifier,
            query_strategy=modAL.dropout.mc_dropout_max_entropy,
        )
        for random_tie_break in [True, False]:
            for num_cycles, sample_per_forward_pass in product(range(1, 5), range(1, 5)):
                for n_samples, n_classes in product(range(1, 5), range(1, 5)):
                    for n_instances in range(1, n_samples):
                        X_pool = torch.randn(n_samples, n_classes)
                        modAL.dropout.mc_dropout_max_entropy(learner, X_pool, n_instances, random_tie_break, [],
                                                             num_cycles, sample_per_forward_pass)

    def test_mc_dropout_max_variationRatios(self):
        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.skorch_classifier,
            query_strategy=modAL.dropout.mc_dropout_max_variationRatios,
        )
        for random_tie_break in [True, False]:
            for num_cycles, sample_per_forward_pass in product(range(1, 5), range(1, 5)):
                for n_samples, n_classes in product(range(1, 5), range(1, 5)):
                    for n_instances in range(1, n_samples):
                        X_pool = torch.randn(n_samples, n_classes)
                        modAL.dropout.mc_dropout_max_variationRatios(learner, X_pool, n_instances, random_tie_break, [],
                                                                     num_cycles, sample_per_forward_pass)

    def test_get_predictions(self):
        X = torch.randn(100, 1)

        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.skorch_classifier,
            query_strategy=mock.MockFunction(return_val=None),
        )

        # num predictions tests
        for num_predictions in range(1, 20):
            for samples_per_forward_pass in range(1, 10):

                predictions = modAL.dropout.get_predictions(
                    learner, X, dropout_layer_indexes=[],
                    num_predictions=num_predictions,
                    sample_per_forward_pass=samples_per_forward_pass)

                self.assertEqual(len(predictions), num_predictions)

        self.assertRaises(AssertionError, modAL.dropout.get_predictions,
                          learner, X, dropout_layer_indexes=[],
                          num_predictions=-1,
                          sample_per_forward_pass=0)

        self.assertRaises(AssertionError, modAL.dropout.get_predictions,
                          learner, X, dropout_layer_indexes=[],
                          num_predictions=10,
                          sample_per_forward_pass=-5)

        # logits adapter function test
        for samples, classes, subclasses in product(range(1, 10),  range(1, 10),  range(1, 10)):
            input_shape = (samples, classes, subclasses)
            desired_shape = (input_shape[0], np.prod(input_shape[1:]))
            X_adaption_needed = torch.randn(input_shape)

            def logits_adaptor(input_tensor, data): return torch.flatten(
                input_tensor, start_dim=1)

            predictions = modAL.dropout.get_predictions(
                learner, X_adaption_needed, dropout_layer_indexes=[],
                num_predictions=num_predictions,
                sample_per_forward_pass=samples_per_forward_pass,
                logits_adaptor=logits_adaptor)

            self.assertEqual(predictions[0].shape, desired_shape)

    def test_set_dropout_mode(self):
        # set dropmout mode for all dropout layers
        for train_mode in [True, False]:
            model = Torch_Model()
            modules = list(model.modules())

            for module in modules:
                self.assertEqual(module.training, True)

            modAL.dropout.set_dropout_mode(model, [], train_mode)

            self.assertEqual(modules[7].training, train_mode)
            self.assertEqual(modules[11].training, train_mode)

        # set dropout mode only for special layers:
        for train_mode in [True, False]:
            model = Torch_Model()
            modules = list(model.modules())
            modAL.dropout.set_dropout_mode(model, [7], train_mode)
            self.assertEqual(modules[7].training, train_mode)
            self.assertEqual(modules[11].training, True)

            modAL.dropout.set_dropout_mode(model, [], True)
            modAL.dropout.set_dropout_mode(model, [11], train_mode)
            self.assertEqual(modules[11].training, train_mode)
            self.assertEqual(modules[7].training, True)

            # No Dropout Layer
            self.assertRaises(KeyError, modAL.dropout.set_dropout_mode,
                              model, [5], train_mode)


class TestDeepActiveLearner(unittest.TestCase):
    """
        Tests for the base class methods of the BaseLearner (base.py) are provided in
        the TestActiveLearner.
    """

    def setUp(self):
        self.mock_deep_estimator = mock.MockEstimator()
        # Add methods that can not be autospecced (because of the wrapper)
        self.mock_deep_estimator.initialize = MagicMock(name='initialize')
        self.mock_deep_estimator.partial_fit = MagicMock(name='partial_fit')

    def test_teach(self):

        for bootstrap, warm_start in product([True, False], [True, False]):
            for n_samples in range(1, 10):
                X = torch.randn(n_samples, 1)
                y = torch.randn(n_samples)

                learner = modAL.models.learners.DeepActiveLearner(
                    estimator=self.mock_deep_estimator
                )

                learner.teach(X, y, bootstrap=bootstrap, warm_start=warm_start)

    def test_batch_size(self):
        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.mock_deep_estimator
        )

        for batch_size in range(1, 50):
            learner.batch_size = batch_size
            self.assertEqual(batch_size, learner.batch_size)

    def test_num_epochs(self):
        learner = modAL.models.learners.DeepActiveLearner(
            estimator=self.mock_deep_estimator
        )

        for num_epochs in range(1, 50):
            learner.num_epochs = num_epochs
            self.assertEqual(num_epochs, learner.num_epochs)


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
                    y_initial = np.random.randint(
                        0, 2, size=(n_samples, n_features+1))
                    y_new = np.random.randint(
                        0, 2, size=(n_new_samples, n_features+1))
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
                    self.assertRaises(
                        ValueError, learner._add_training_data, X_new, y_new)
                    # 2. X_new has wrong dimensions
                    X_new = np.random.rand(n_new_samples, 2*n_features)
                    y_new = np.random.randint(0, 2, size=(n_new_samples,))
                    self.assertRaises(
                        ValueError, learner._add_training_data, X_new, y_new)

    def test_predict(self):
        for n_samples in range(1, 100):
            for n_features in range(1, 10):
                X = np.random.rand(n_samples, n_features)
                predict_return = np.random.randint(0, 2, size=(n_samples, ))
                mock_classifier = mock.MockEstimator(
                    predict_return=predict_return)
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
                predict_proba_return = np.random.randint(
                    0, 2, size=(n_samples,))
                mock_classifier = mock.MockEstimator(
                    predict_proba_return=predict_proba_return)
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
                query_metrics = np.random.randint(0, n_samples)
                mock_query = mock.MockFunction(
                    return_val=(query_idx, query_metrics))
                learner = modAL.models.learners.ActiveLearner(
                    estimator=None,
                    query_strategy=mock_query
                )
                np.testing.assert_equal(
                    learner.query(X),
                    (query_idx, X[query_idx])
                )
                np.testing.assert_equal(
                    learner.query(X, return_metrics=True),
                    (query_idx, X[query_idx], query_metrics)
                )

    def test_score(self):
        test_cases = (np.random.rand() for _ in range(10))
        for score_return in test_cases:
            mock_classifier = mock.MockEstimator(score_return=score_return)
            learner = modAL.models.learners.ActiveLearner(
                mock_classifier, mock.MockFunction(None))
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

    def test_nan(self):
        X_training_nan = np.ones(shape=(10, 2)) * np.nan
        X_training_inf = np.ones(shape=(10, 2)) * np.inf
        y_training = np.random.randint(0, 2, size=10)

        learner = modAL.models.learners.ActiveLearner(
            X_training=X_training_nan, y_training=y_training,
            estimator=mock.MockEstimator(),
            force_all_finite=False
        )
        learner.teach(X_training_nan, y_training)

        learner = modAL.models.learners.ActiveLearner(
            X_training=X_training_inf, y_training=y_training,
            estimator=mock.MockEstimator(),
            force_all_finite=False
        )
        learner.teach(X_training_inf, y_training)

    def test_keras(self):
        pass

    def test_sklearn(self):
        learner = modAL.models.learners.ActiveLearner(
            estimator=RandomForestClassifier(n_estimators=10),
            X_training=np.random.rand(10, 10),
            y_training=np.random.randint(0, 2, size=(10,))
        )
        learner.fit(np.random.rand(10, 10),
                    np.random.randint(0, 2, size=(10,)))
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
            initial_idx = np.random.choice(
                range(n_samples), size=5, replace=False)

            learner = modAL.models.learners.ActiveLearner(
                estimator=RandomForestClassifier(n_estimators=10), query_strategy=query_strategy,
                X_training=X_pool[initial_idx], y_training=y_pool[initial_idx]
            )
            query_idx, query_inst = learner.query(X_pool)
            learner.teach(X_pool[query_idx], y_pool[query_idx])

    def test_on_transformed(self):
        n_samples = 10
        n_features = 5
        query_strategies = [
            modAL.batch.uncertainty_batch_sampling
            # add further strategies which work with instance representations
            # no further ones as of 25.09.2020
        ]
        X_pool = np.random.rand(n_samples, n_features)

        # use pandas data frame as X_pool, which will be transformed back to numpy with sklearn pipeline
        X_pool = pd.DataFrame(X_pool)

        y_pool = np.random.randint(0, 2, size=(n_samples,))
        train_idx = np.random.choice(range(n_samples), size=2, replace=False)

        for query_strategy in query_strategies:
            learner = modAL.models.learners.ActiveLearner(
                estimator=make_pipeline(
                    FunctionTransformer(func=pd.DataFrame.to_numpy),
                    RandomForestClassifier(n_estimators=10)
                ),
                query_strategy=query_strategy,
                X_training=X_pool.iloc[train_idx],
                y_training=y_pool[train_idx],
                on_transformed=True
            )
            query_idx, query_inst = learner.query(X_pool)
            learner.teach(X_pool.iloc[query_idx], y_pool[query_idx])

    def test_on_transformed_with_variable_transformation(self):
        """
        Learnable transformations naturally change after a model is retrained. Make sure this is handled
        properly for on_transformed=True query strategies.
        """
        query_strategies = [
            modAL.batch.uncertainty_batch_sampling
            # add further strategies which work with instance representations
            # no further ones as of 09.12.2020
        ]

        X_labeled = ['Dog', 'Cat', 'Tree']

        # contains unseen in labeled words, training model on those
        # will alter CountVectorizer transformations
        X_pool = ['Airplane', 'House']

        y = [0, 1, 1, 0, 1]  # irrelevant for test

        for query_strategy in query_strategies:
            learner = modAL.models.learners.ActiveLearner(
                estimator=make_pipeline(
                    CountVectorizer(),
                    RandomForestClassifier(n_estimators=10)
                ),
                query_strategy=query_strategy,
                X_training=X_labeled, y_training=y[:len(X_labeled)],
                on_transformed=True,
            )

            for _ in range(len(X_pool)):
                query_idx, query_instance = learner.query(
                    X_pool, n_instances=1)
                i = query_idx[0]

                learner.teach(
                    X=[X_pool[i]],
                    y=[y[i]]
                )

    def test_old_query_strategy_interface(self):
        n_samples = 10
        n_features = 5
        X_pool = np.random.rand(n_samples, n_features)
        y_pool = np.random.randint(0, 2, size=(n_samples,))

        # defining a custom query strategy also returning the selected instance
        # make sure even if a query strategy works in some funny way
        # (e.g. instance not matching instance index),
        # the old interface remains unchanged
        query_idx_ = np.random.choice(n_samples, 2)
        query_instance_ = X_pool[query_idx_]

        def custom_query_strategy(classifier, X):
            return query_idx_, query_instance_

        train_idx = np.random.choice(range(n_samples), size=2, replace=False)
        custom_query_learner = modAL.models.learners.ActiveLearner(
            estimator=RandomForestClassifier(n_estimators=10),
            query_strategy=custom_query_strategy,
            X_training=X_pool[train_idx], y_training=y_pool[train_idx]
        )

        query_idx, query_instance = custom_query_learner.query(X_pool)
        custom_query_learner.teach(
            X=X_pool[query_idx],
            y=y_pool[query_idx]
        )
        np.testing.assert_equal(query_idx, query_idx_)
        np.testing.assert_equal(query_instance, query_instance_)


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
                learner = modAL.models.learners.BayesianOptimizer(
                    estimator=regressor)
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
                optimizer = modAL.models.learners.BayesianOptimizer(
                    regressor, X_training=X, y_training=y)
                X_max, y_max = optimizer.get_max()
                np.testing.assert_equal(X_max, X[max_idx])
                np.testing.assert_equal(y_max, y[max_idx])

    def test_teach(self):
        for bootstrap, only_new in product([True, False], [True, False]):
            # case 1. optimizer is uninitialized
            for n_samples in range(1, 100):
                for n_features in range(1, 100):
                    regressor = mock.MockEstimator()
                    learner = modAL.models.learners.BayesianOptimizer(
                        estimator=regressor)

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

    def test_on_transformed(self):
        n_samples = 10
        n_features = 5
        query_strategies = [
            # TODO remove, added just to make sure on_transformed doesn't break anything
            # but it has no influence on this strategy, nothing special tested here
            mock.MockFunction(return_val=[np.random.randint(0, n_samples)])

            # add further strategies which work with instance representations
            # no further ones as of 25.09.2020
        ]
        X_pool = np.random.rand(n_samples, n_features)

        # use pandas data frame as X_pool, which will be transformed back to numpy with sklearn pipeline
        X_pool = pd.DataFrame(X_pool)

        y_pool = np.random.rand(n_samples)
        train_idx = np.random.choice(range(n_samples), size=2, replace=False)

        for query_strategy in query_strategies:
            learner = modAL.models.learners.BayesianOptimizer(
                estimator=make_pipeline(
                    FunctionTransformer(func=pd.DataFrame.to_numpy),
                    GaussianProcessRegressor()
                ),
                query_strategy=query_strategy,
                X_training=X_pool.iloc[train_idx],
                y_training=y_pool[train_idx],
                on_transformed=True
            )
            query_idx, query_inst = learner.query(X_pool)
            learner.teach(X_pool.iloc[query_idx], y_pool[query_idx])


class TestCommittee(unittest.TestCase):

    def test_set_classes(self):
        # 1. test unfitted learners
        for n_learners in range(1, 10):
            learner_list = [modAL.models.learners.ActiveLearner(estimator=mock.MockEstimator(fitted=False))
                            for idx in range(n_learners)]
            committee = modAL.models.learners.Committee(
                learner_list=learner_list)
            self.assertEqual(committee.classes_, None)
            self.assertEqual(committee.n_classes_, 0)

        # 2. test fitted learners
        for n_classes in range(1, 10):
            learner_list = [modAL.models.learners.ActiveLearner(estimator=mock.MockEstimator(classes_=np.asarray([idx])))
                            for idx in range(n_classes)]
            committee = modAL.models.learners.Committee(
                learner_list=learner_list)
            np.testing.assert_equal(
                committee.classes_,
                np.unique(range(n_classes))
            )

    def test_predict(self):
        for n_learners in range(1, 10):
            for n_instances in range(1, 10):
                prediction = np.random.randint(
                    10, size=(n_instances, n_learners))
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
                    vote_proba_output = np.random.rand(
                        n_samples, n_learners, n_classes)
                    # assembling the mock learners
                    learner_list = [mock.MockActiveLearner(
                        predict_proba_return=vote_proba_output[:,
                                                               learner_idx, :],
                        predictor=mock.MockEstimator(
                            classes_=list(range(n_classes)))
                    ) for learner_idx in range(n_learners)]
                    committee = modAL.models.learners.Committee(
                        learner_list=learner_list)
                    np.testing.assert_almost_equal(
                        committee.predict_proba(np.random.rand(n_samples, 1)),
                        np.mean(vote_proba_output, axis=1)
                    )

    def test_vote(self):
        for n_members in range(1, 10):
            for n_instances in range(1, 100):
                vote_output = np.random.randint(
                    0, 2, size=(n_instances, n_members))
                # assembling the Committee
                learner_list = [mock.MockActiveLearner(
                    predict_return=vote_output[:, member_idx],
                    predictor=mock.MockEstimator(classes_=[0])
                )
                    for member_idx in range(n_members)]
                committee = modAL.models.learners.Committee(
                    learner_list=learner_list)
                np.testing.assert_array_almost_equal(
                    committee.vote(np.random.rand(n_instances).reshape(-1, 1)),
                    vote_output
                )

    def test_vote_proba(self):
        for n_samples in range(1, 100):
            for n_learners in range(1, 10):
                for n_classes in range(1, 10):
                    vote_proba_output = np.random.rand(
                        n_samples, n_learners, n_classes)
                    # assembling the mock learners
                    learner_list = [mock.MockActiveLearner(
                        predict_proba_return=vote_proba_output[:,
                                                               learner_idx, :],
                        predictor=mock.MockEstimator(
                            classes_=list(range(n_classes)))
                    ) for learner_idx in range(n_learners)]
                    committee = modAL.models.learners.Committee(
                        learner_list=learner_list)
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

    def test_on_transformed(self):
        n_samples = 10
        n_features = 5
        query_strategies = [
            modAL.batch.uncertainty_batch_sampling
            # add further strategies which work with instance representations
            # no further ones as of 25.09.2020
        ]
        X_pool = np.random.rand(n_samples, n_features)

        # use pandas data frame as X_pool, which will be transformed back to numpy with sklearn pipeline
        X_pool = pd.DataFrame(X_pool)

        y_pool = np.random.randint(0, 2, size=(n_samples,))
        train_idx = np.random.choice(range(n_samples), size=5, replace=False)

        learner_list = [modAL.models.learners.ActiveLearner(
            estimator=make_pipeline(
                FunctionTransformer(func=pd.DataFrame.to_numpy),
                RandomForestClassifier(n_estimators=10)
            ),
            # committee learners can contain different amounts of
            # different instances
            X_training=X_pool.iloc[train_idx[(
                np.arange(i + 1) + i) % len(train_idx)]],
            y_training=y_pool[train_idx[(
                np.arange(i + 1) + i) % len(train_idx)]],
        ) for i in range(3)]

        for query_strategy in query_strategies:
            committee = modAL.models.learners.Committee(
                learner_list=learner_list,
                query_strategy=query_strategy,
                on_transformed=True
            )
            query_idx, query_inst = committee.query(X_pool)
            committee.teach(X_pool.iloc[query_idx], y_pool[query_idx])


class TestCommitteeRegressor(unittest.TestCase):

    def test_predict(self):
        for n_members in range(1, 10):
            for n_instances in range(1, 100):
                vote = np.random.rand(n_instances, n_members)
                # assembling the Committee
                learner_list = [mock.MockActiveLearner(predict_return=vote[:, member_idx])
                                for member_idx in range(n_members)]
                committee = modAL.models.learners.CommitteeRegressor(
                    learner_list=learner_list)
                np.testing.assert_array_almost_equal(
                    committee.predict(np.random.rand(
                        n_instances).reshape(-1, 1), return_std=False),
                    np.mean(vote, axis=1)
                )
                np.testing.assert_array_almost_equal(
                    committee.predict(np.random.rand(
                        n_instances).reshape(-1, 1), return_std=True),
                    (np.mean(vote, axis=1), np.std(vote, axis=1))
                )

    def test_vote(self):
        for n_members in range(1, 10):
            for n_instances in range(1, 100):
                vote_output = np.random.rand(n_instances, n_members)
                # assembling the Committee
                learner_list = [mock.MockActiveLearner(predict_return=vote_output[:, member_idx])
                                for member_idx in range(n_members)]
                committee = modAL.models.learners.CommitteeRegressor(
                    learner_list=learner_list)
                np.testing.assert_array_almost_equal(
                    committee.vote(np.random.rand(n_instances).reshape(-1, 1)),
                    vote_output
                )

    def test_on_transformed(self):
        n_samples = 10
        n_features = 5
        query_strategies = [
            # TODO remove, added just to make sure on_transformed doesn't break anything
            # but it has no influence on this strategy, nothing special tested here
            mock.MockFunction(return_val=[np.random.randint(0, n_samples)])

            # add further strategies which work with instance representations
            # no further ones as of 25.09.2020
        ]
        X_pool = np.random.rand(n_samples, n_features)

        # use pandas data frame as X_pool, which will be transformed back to numpy with sklearn pipeline
        X_pool = pd.DataFrame(X_pool)

        y_pool = np.random.rand(n_samples)
        train_idx = np.random.choice(range(n_samples), size=2, replace=False)

        learner_list = [modAL.models.learners.ActiveLearner(
            estimator=make_pipeline(
                FunctionTransformer(func=pd.DataFrame.to_numpy),
                GaussianProcessRegressor()
            ),
            # committee learners can contain different amounts of
            # different instances
            X_training=X_pool.iloc[train_idx[(
                np.arange(i + 1) + i) % len(train_idx)]],
            y_training=y_pool[train_idx[(
                np.arange(i + 1) + i) % len(train_idx)]],
        ) for i in range(3)]

        for query_strategy in query_strategies:
            committee = modAL.models.learners.CommitteeRegressor(
                learner_list=learner_list,
                query_strategy=query_strategy,
                on_transformed=True
            )
            query_idx, query_inst = committee.query(X_pool)
            committee.teach(X_pool.iloc[query_idx], y_pool[query_idx])


class TestMultilabel(unittest.TestCase):
    def test_SVM_loss(self):
        for n_classes in range(2, 10):
            for n_instances in range(1, 10):
                X_training = np.random.rand(n_instances, 5)
                y_training = np.random.randint(
                    0, 2, size=(n_instances, n_classes))
                X_pool = np.random.rand(n_instances, 5)
                y_pool = np.random.randint(0, 2, size=(n_instances, n_classes))
                classifier = OneVsRestClassifier(
                    SVC(probability=True, gamma='auto'))
                classifier.fit(X_training, y_training)
                avg_loss = modAL.multilabel._SVM_loss(classifier, X_pool)
                mcc_loss = modAL.multilabel._SVM_loss(classifier, X_pool,
                                                      most_certain_classes=np.random.randint(0, n_classes, size=(n_instances)))
                self.assertEqual(avg_loss.shape, (len(X_pool), ))
                self.assertEqual(mcc_loss.shape, (len(X_pool),))

    def test_strategies(self):
        for n_classes in range(3, 10):
            for n_pool_instances in range(1, 10):
                for n_query_instances in range(1, min(n_pool_instances, 3)):
                    X_training = np.random.rand(n_pool_instances, 5)
                    y_training = np.random.randint(
                        0, 2, size=(n_pool_instances, n_classes))
                    X_pool = np.random.rand(n_pool_instances, 5)
                    classifier = OneVsRestClassifier(
                        SVC(probability=True, gamma='auto'))
                    classifier.fit(X_training, y_training)

                    active_learner = modAL.models.ActiveLearner(classifier)
                    # no random tie break
                    modAL.multilabel.SVM_binary_minimum(active_learner, X_pool)
                    modAL.multilabel.mean_max_loss(
                        classifier, X_pool, n_query_instances)
                    modAL.multilabel.max_loss(
                        classifier, X_pool, n_query_instances)
                    modAL.multilabel.min_confidence(
                        classifier, X_pool, n_query_instances)
                    modAL.multilabel.avg_confidence(
                        classifier, X_pool, n_query_instances)
                    modAL.multilabel.max_score(
                        classifier, X_pool, n_query_instances)
                    modAL.multilabel.avg_score(
                        classifier, X_pool, n_query_instances)
                    # random tie break
                    modAL.multilabel.SVM_binary_minimum(
                        active_learner, X_pool, random_tie_break=True)
                    modAL.multilabel.mean_max_loss(
                        classifier, X_pool, n_query_instances, random_tie_break=True)
                    modAL.multilabel.max_loss(
                        classifier, X_pool, n_query_instances, random_tie_break=True)
                    modAL.multilabel.min_confidence(
                        classifier, X_pool, n_query_instances, random_tie_break=True)
                    modAL.multilabel.avg_confidence(
                        classifier, X_pool, n_query_instances, random_tie_break=True)
                    modAL.multilabel.max_score(
                        classifier, X_pool, n_query_instances, random_tie_break=True)
                    modAL.multilabel.avg_score(
                        classifier, X_pool, n_query_instances, random_tie_break=True)


class TestExamples(unittest.TestCase):

    def test_examples(self):
        import example_tests.active_regression
        import example_tests.bagging
        import example_tests.bayesian_optimization
        import example_tests.custom_query_strategies
        import example_tests.ensemble
        import example_tests.ensemble_regression
        import example_tests.information_density
        import example_tests.multidimensional_data
        import example_tests.pool_based_sampling
        import example_tests.query_by_committee
        import example_tests.ranked_batch_mode
        import example_tests.shape_learning
        import example_tests.stream_based_sampling


if __name__ == '__main__':
    unittest.main(verbosity=2)
