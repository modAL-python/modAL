class MockClassifier:
    def __init__(self, predict_proba_return):
        self.predict_proba_return = predict_proba_return

    def predict_proba(self, data):
        return self.predict_proba_return
