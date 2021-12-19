from time import time

import numpy as np
from acton.acton import main as acton_main
from alp.active_learning.active_learning import \
    ActiveLearner as ActiveLearnerALP
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.models.logistic_regression import \
    LogisticRegression as LogisticRegressionLibact
from libact.query_strategies import QueryByCommittee, UncertaintySampling
from libact.query_strategies.multiclass.expected_error_reduction import EER
from modAL.expected_error import expected_error_reduction
from modAL.models import ActiveLearner, Committee
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

runtime = {}


def timeit(n_reps=10):

    def timer(func):

        def timed_func(*args, **kwargs):
            start = time()
            for _ in range(n_reps):
                result = func(*args, **kwargs)
            end = time()
            print("%s has been executed in %f s avg for %d reps" % (func.__name__, (end - start)/n_reps, n_reps))
            runtime[func.__name__] = (end - start)/n_reps
            return result

        return timed_func

    return timer


@timeit()
def libact_uncertainty(X, y, n_queries):
    y_train = np.array([None for _ in range(len(y))])
    y_train[0], y_train[50], y_train[100] = 0, 1, 2
    libact_train_dataset = Dataset(X, y_train)
    libact_full_dataset = Dataset(X, y)
    libact_learner = LogisticRegressionLibact(solver='liblinear', n_jobs=1, multi_class='ovr') #SVM(gamma='auto', probability=True)
    libact_qs = UncertaintySampling(libact_train_dataset, model=libact_learner, method='lc')
    libact_labeler = IdealLabeler(libact_full_dataset)
    libact_learner.train(libact_train_dataset)

    for _ in range(n_queries):
        query_idx = libact_qs.make_query()
        query_label = libact_labeler.label(X[query_idx])
        libact_train_dataset.update(query_idx, query_label)
        libact_learner.train(libact_train_dataset)


@timeit()
def libact_EER(X, y, n_queries):
    y_train = np.array([None for _ in range(len(y))])
    y_train[0], y_train[50], y_train[100] = 0, 1, 2
    libact_train_dataset = Dataset(X, y_train)
    libact_full_dataset = Dataset(X, y)
    libact_learner = LogisticRegressionLibact(solver='liblinear', n_jobs=1, multi_class='ovr') #SVM(gamma='auto', probability=True)
    libact_qs = EER(libact_train_dataset, model=libact_learner, loss='01')
    libact_labeler = IdealLabeler(libact_full_dataset)
    libact_learner.train(libact_train_dataset)

    for _ in range(n_queries):
        query_idx = libact_qs.make_query()
        query_label = libact_labeler.label(X[query_idx])
        libact_train_dataset.update(query_idx, query_label)
        libact_learner.train(libact_train_dataset)


@timeit()
def libact_QBC(X, y, n_queries):
    y_train = np.array([None for _ in range(len(y))])
    y_train[0], y_train[50], y_train[100] = 0, 1, 2
    libact_train_dataset = Dataset(X, y_train)
    libact_full_dataset = Dataset(X, y)
    libact_learner_list = [LogisticRegressionLibact(solver='liblinear', n_jobs=1, multi_class='ovr'),
                           LogisticRegressionLibact(solver='liblinear', n_jobs=1, multi_class='ovr')]
    libact_qs = QueryByCommittee(libact_train_dataset, models=libact_learner_list,
                                 method='lc')
    libact_labeler = IdealLabeler(libact_full_dataset)
    for libact_learner in libact_learner_list:
        libact_learner.train(libact_train_dataset)

    for _ in range(n_queries):
        query_idx = libact_qs.make_query()
        query_label = libact_labeler.label(X[query_idx])
        libact_train_dataset.update(query_idx, query_label)
        for libact_learner in libact_learner_list:
            libact_learner.train(libact_train_dataset)


@timeit()
def modAL_uncertainty(X, y, n_queries):
    modAL_learner = ActiveLearner(LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr'),
                                  X_training=X[[0, 50, 100]], y_training=y[[0, 50, 100]])

    for _ in range(n_queries):
        query_idx, query_inst = modAL_learner.query(X)
        modAL_learner.teach(X[query_idx], y[query_idx])


@timeit()
def modAL_QBC(X, y, n_queries):
    learner_list = [ActiveLearner(LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr'),
                                  X_training=X[[0, 50, 100]], y_training=y[[0, 50, 100]]),
                    ActiveLearner(LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr'),
                                  X_training=X[[0, 50, 100]], y_training=y[[0, 50, 100]])]

    modAL_learner = Committee(learner_list)

    for _ in range(n_queries):
        query_idx, query_inst = modAL_learner.query(X)
        modAL_learner.teach(X[query_idx], y[query_idx])


@timeit()
def modAL_EER(X, y, n_queries):
    modAL_learner = ActiveLearner(LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr'),
                                  query_strategy=expected_error_reduction,
                                  X_training=X[[0, 50, 100]], y_training=y[[0, 50, 100]])

    for _ in range(n_queries):
        query_idx, query_inst = modAL_learner.query(X)
        modAL_learner.teach(X[query_idx], y[query_idx])


@timeit()
# acton requires a txt format for data
def acton_uncertainty(data_path, n_queries):
    # acton has no SVM support, so the LogisticRegression model is used
    acton_main(
        data_path=data_path,
        feature_cols=['feat01', 'feat02', 'feat03', 'feat04'],
        label_col='label',
        output_path='out.csv',
        n_epochs=n_queries,
        initial_count=3,
        recommender='UncertaintyRecommender',
        predictor='LogisticRegression')


@timeit()
# acton requires a txt format for data
def acton_QBC(data_path, n_queries):
    # acton has no SVM support, so the LogisticRegression model is used
    acton_main(
        data_path=data_path,
        feature_cols=['feat01', 'feat02', 'feat03', 'feat04'],
        label_col='label',
        output_path='out.csv',
        n_epochs=n_queries,
        initial_count=3,
        recommender='QBCRecommender',
        predictor='LogisticRegressionCommittee')


@timeit()
def alp_uncertainty(X, y, n_queries):
    X_labeled, y_labeled = X[[0, 50, 100]], y[[0, 50, 100]]
    estimator = LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr')
    estimator.fit(X_labeled, y_labeled)
    learner = ActiveLearnerALP(strategy='least_confident')

    for _ in range(n_queries):
        query_idx = learner.rank(estimator, X, num_queries=1)
        X_labeled = np.concatenate((X_labeled, X[query_idx]), axis=0)
        y_labeled = np.concatenate((y_labeled, y[query_idx]), axis=0)
        estimator.fit(X_labeled, y_labeled)


@timeit()
def alp_QBC(X, y, n_queries):
    X_labeled, y_labeled = X[[0, 50, 100]], y[[0, 50, 100]]
    estimators = [LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr'),
                  LogisticRegression(solver='liblinear', n_jobs=1, multi_class='ovr')]

    for estimator in estimators:
        estimator.fit(X_labeled, y_labeled)

    learner = ActiveLearnerALP(strategy='vote_entropy')

    for _ in range(n_queries):
        query_idx = learner.rank(estimators, X, num_queries=1)
        X_labeled = np.concatenate((X_labeled, X[query_idx]), axis=0)
        y_labeled = np.concatenate((y_labeled, y[query_idx]), axis=0)
        for estimator in estimators:
            estimator.fit(X_labeled, y_labeled)


def comparisons(n_queries=10):
    # loading the data
    X, y = load_iris(return_X_y=True)

    libact_uncertainty(X, y, n_queries)
    libact_QBC(X, y, n_queries)
    libact_EER(X, y, n_queries)
    acton_uncertainty('iris.csv', n_queries)
    acton_QBC('iris.csv', n_queries)
    alp_uncertainty(X, y, n_queries)
    alp_QBC(X, y, n_queries)
    modAL_uncertainty(X, y, n_queries)
    modAL_QBC(X, y, n_queries)
    modAL_EER(X, y, n_queries)


if __name__ == '__main__':
    comparisons()
    print(runtime)
