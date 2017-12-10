class MockUtility:
    """
    Mock utility function for testing.
    """
    def __init__(self, utility_return):
        self.utility_return = utility_return

    def __call__(self, *args):
        return self.utility_return


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

    def predict(self, data):
        return self.predict_return

    def predict_proba(self, data):
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
            classes_=None,
    ):
        self.predictor = predictor
        self.uncertainty_measure = uncertainty_measure
        self.query_strategy = query_strategy

        self.predict_proba_return = predict_proba_return
        self.calculate_utility_return = calculate_utility_return
        self.predict_return = predict_return
        self.score_return = score_return

        self.classes_ = classes_

    def calculate_uncertainty(self, data):
        return self.calculate_utility_return

    def fit(self, *args, **kwargs):
        pass

    def predict(self, data):
        return self.predict_return

    def predict_proba(self, data):
        return self.predict_proba_return

    def score(self, X, y, sample_weight=None):
        return self.score_return