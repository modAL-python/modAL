class MockClassifier:
    """
    Mock classifier object for testing. The predict_proba method returns the
    object given for argument predict_proba_return.
    """
    def __init__(
            self, predict_proba_return=None, calculate_utility_return=None, predict_return=None
    ):
        self.predict_proba_return = predict_proba_return
        self.calculate_utility_return = calculate_utility_return
        self.predict_return = None

    def predict_proba(self, data):
        return self.predict_proba_return

    def predict(self, data):
        return self.predict_return

    def calculate_utility(self, data):
        return self.calculate_utility_return
