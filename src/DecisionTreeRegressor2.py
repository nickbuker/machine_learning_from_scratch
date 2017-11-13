import numpy as np
from Node import Node
from scoring import R2, RSS

class DecisionTreeRegressor:
    def __init__(self):
        self.tree = Node()

    @_take_parameters
    def fit(self, X, y, max_depth):
        X = self._make_numpy(data=X)
        y = self._make_numpy(data=y)
        # TODO continue implementation

    def predict(self, X):
        X =  self._make_numpy(data=X)
        return np.apply_along_axis(func1d=self._query_tree, axis=1, arr=X)

    def score(self, X, y):
        X = self._make_numpy(data=X)
        y = self._make_numpy(data=y)
        y_hat = self.predict(X)
        return R2(y, y_hat)

    def _take_parameters(self, X, y, max_depth):
        def wrap(fn):
            kwargs = {'X': X, 'y': y, 'max_depth': max_depth}
            return fn(**kwargs)
        return wrap

    def _make_numpy(self, data):
        if not isinstance(data, np.ndarray):
            np.array(data)
        return data

    def _find_best_col(self, X, y):
        error = np.inf
        col = 0
        split = 0
        b_mean = 0
        a_mean = 0
        for i in range(0, X.shape[1]):
            temp_error, temp_split, temp_b_mean, temp_a_mean = self._find_best_split(X[:, i], y)
            if temp_error < error:
                error = temp_error
                col = i
                split = temp_split
                b_mean = temp_b_mean
                a_mean = temp_a_mean
        return col, split, b_mean, a_mean

    def _find_best_split(self, col_values, y):
        error = np.inf
        split = 0
        b_mean = 0
        a_mean = 0
        for n in np.unique(col_values):
            mask = col_values <= n
            temp_b_mean = np.mean(y[mask])
            temp_a_mean = np.mean(y[np.invert(mask)])
            y_hat = np.array(sum(mask) * [temp_b_mean] + sum(np.invert(mask)) * [temp_a_mean])
            temp_error = RSS(y, y_hat)
            if temp_error < error:
                error = temp_error
                split = n
                b_mean = temp_b_mean
                a_mean = temp_a_mean
        return error, split, b_mean, a_mean

    def _query_tree(self, row):
        return self.tree.get_leaf(row=row)