class Model:
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError
