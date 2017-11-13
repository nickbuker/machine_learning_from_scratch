import numpy as np
from Model import Model

class DecisionTreeRegressor(Model):
    def __init__(self):
        super(Model, self).__init__()

    def fit(self, X, y):
        X = self._make_numpy(X)
        y = self._make_numpy(y)

    def predict(self, X):
        X =  self._make_numpy(X)

    def score(self, X, y):
        X = self._make_numpy(X)
        y = self._make_numpy(y)

    def _make_numpy(self, data):
        if not isinstance(data, np.ndarray):
            np.array(data)
        return data

    def _find_best_col(self, X, y):
        for i in range(0, X.shape[1]):
            self._find_best_split(X[:, i], y)

    def _find_best_split(self, col_values, y):
        error = np.inf
        split = None
        y_hat = None
        for n in np.unique(col_values):
            mask = col_values <= n
            b_mean = np.mean(y[mask])
            a_mean = np.mean(y[np.invert(mask)])
            temp_y_hat = np.array(sum(mask) * [b_mean] + sum(np.invert(mask)) * [a_mean])
            temp_error = sum((y - temp_y_hat) ** 2)
            if temp_error < error:
                error = temp_error
                split = n
                y_hat = temp_y_hat
