from sklearn.exceptions import NotFittedError


class MockFunction:
    """
    Mock utility function for testing.
    """
    def __init__(self, return_val):
        self.return_val = return_val

    def __call__(self, *args):
        return self.return_val


class MockEstimator:
    """
    Mock classifier object for testing.
    """
    def __init__(
            self, predict_proba_return=None, predict_return=None, score_return=None,
            classes_=None, fitted=True
    ):
        self.fitted = fitted

        if fitted:
            self.classes_ = classes_

        self.predict_return = predict_return
        self.predict_proba_return = predict_proba_return
        self.score_return = score_return

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError

        return self.predict_return

    def predict_proba(self, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError

        return self.predict_proba_return

    def score(self, *args, **kwargs):
        return self.score_return


class MockActiveLearner:
    """
    Mock ActiveLearner for testing.
    """
    def __init__(
            self, predictor=None, query_strategy=None,
            predict_proba_return=None, calculate_utility_return=None, predict_return=None, score_return=None,
            _X_initial=None, _y_initial=None
    ):
        self.estimator = predictor
        self.query_strategy = query_strategy

        self.predict_proba_return = predict_proba_return
        self.calculate_utility_return = calculate_utility_return
        self.predict_return = predict_return
        self.score_return = score_return

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return self.predict_return

    def predict_proba(self, *args, **kwargs):
        return self.predict_proba_return

    def score(self, *args, **kwargs):
        return self.score_return


class MockCommittee:
    """
    Mock Committee for testing.
    """
    def __init__(
            self, n_learners=1, classes_=None, fitted=True,
            calculate_disagreement_return=None,
            predict_return=None, predict_proba_return=None,
            vote_return=None, vote_proba_return=None
    ):
        self.fitted = fitted
        self.n_learners = n_learners

        if fitted:
            self.classes_ = classes_
        else:
            self.classes_ = None

        self.calculate_disagreement_return = calculate_disagreement_return
        self.predict_return = predict_return
        self.predict_proba_return = predict_proba_return
        self.vote_return = vote_return
        self.vote_proba_return = vote_proba_return

    def __len__(self):
        return self.n_learners

    def __iter__(self):
        for x in range(self.n_learners):
            yield x

    def _calculate_disagreement(self, *args, **kwargs):
        return self.calculate_disagreement_return

    def predict(self, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError

        return self.predict_return

    def predict_proba(self, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError

        return self.predict_proba_return

    def vote(self, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError

        return self.vote_return

    def vote_proba(self, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError

        return self.vote_proba_return
