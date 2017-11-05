import numpy as np
# from sklearn.? import bootstrap
from scoring import R2, RSS
from DecisionTreeRegressor import DecisionTreeRegressor


class RandomForestRegressor:

    def __init__(self):
        pass

    def fit(self, X, y, max_depth, n_estimators):
        # instantiate forest of decision trees
        self.trees = [DecisionTreeRegressor() for _ in range(n_estimators)]
        # create masks for rows and columns of data for each tree
        self.rows = [self._sample_rows(X.index) for _ in range(n_estimators)]
        n_cols = int(len(X.columns) ** 0.5)
        self.cols = [self._sample_cols(X.columns, n_cols) for _ in range(n_estimators)]
        for i, tree in enumerate(self.trees):
            tree.fit(X[self.rows[i]][self.cols[i]], y[self.rows[i]])

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def _sample_rows(self, index):
        pass

    def _sample_cols(self, cols, n_cols):
        pass

