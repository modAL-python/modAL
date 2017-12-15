class MockFunction:
    """
    Mock utility function for testing.
    """
    def __init__(self, return_val):
        self.return_val = return_val

    def __call__(self, *args):
        return self.return_val


class MockClassifier:
    """
    Mock classifier object for testing.
    """
    def __init__(
            self, predict_proba_return=None, predict_return=None, score_return=None,
            classes_=None
    ):
        self.classes_ = classes_

        self.predict_return = predict_return
        self.predict_proba_return = predict_proba_return
        self.score_return = score_return

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        return self.predict_return

    def predict_proba(self, X):
        return self.predict_proba_return

    def score(self, X, y, sample_weight=None):
        return self.score_return


class MockActiveLearner:
    """
    Mock ActiveLearner for testing.
    """
    def __init__(
            self, predictor=None, uncertainty_measure=None, query_strategy=None,
            predict_proba_return=None, calculate_utility_return=None, predict_return=None, score_return=None,
            _X_initial=None, _y_initial=None
    ):
        self._predictor = predictor
        self.uncertainty_measure = uncertainty_measure
        self.query_strategy = query_strategy

        self.predict_proba_return = predict_proba_return
        self.calculate_utility_return = calculate_utility_return
        self.predict_return = predict_return
        self.score_return = score_return

    def calculate_uncertainty(self, X):
        return self.calculate_utility_return

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        return self.predict_return

    def predict_proba(self, X):
        return self.predict_proba_return

    def score(self, X, y, sample_weight=None):
        return self.score_return


class MockCommittee:
    """
    Mock Committee for testing.
    """
    def __init__(self, calculate_disagreement_return=None):
        self.calculate_disagreement_return = calculate_disagreement_return

    def calculate_disagreement(self, X):
        return self.calculate_disagreement_return
