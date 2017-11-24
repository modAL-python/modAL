class MockUtility:
    def __init__(self, utility_return):
        self.utility_return = utility_return

    def __call__(self, *args):
        return self.utility_return


class MockClassifier:
    """
    Mock classifier object for testing. The predict_proba method returns the
    object given for argument predict_proba_return.
    """
    def __init__(
            self, predict_proba_return=None, calculate_utility_return=None, predict_return=None,
            score_return=None
    ):
        self.calculate_utility_return = calculate_utility_return
        self.predict_return = predict_return
        self.predict_proba_return = predict_proba_return
        self.score_return = score_return

    def calculate_utility(self, data):
        return self.calculate_utility_return

    def predict(self, data):
        return self.predict_return

    def predict_proba(self, data):
        return self.predict_proba_return

    def score(self, data):
        return self.score_return
