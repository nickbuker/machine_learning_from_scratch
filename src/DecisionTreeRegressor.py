import numpy as np
import pandas as pd
from scoring import R2
from Tree import Tree


class DecisionTreeRegressor:

    def __init__(self):
        pass

    def fit(self, X, y, max_depth):
        self.tree = Tree()
        self.rules = []
        X = self._check_x_data_type(X)
        X['y'] = y
        X['y_hat'] = np.mean(y)
        X = X.sort_values(by='y')
        X = X.reset_index(drop=True)
        data_cols = [col for col in X.columns if col not in ['y', 'y_hat']]
        for _ in range(max_depth):
            score = [0, data_cols[0], -1]
            temp_y_hat
            for col in data_cols:
                for j, element in enumerate(X[col]):
                    below = X[X[col] <= element]
                    above = X[X[col] > element]
                    if len(above) == 0:
                        continue


    def predict(self, X):
        X = self._check_x_data_type(X)

    def score(self, y, y_hat):
        return R2(y, y_hat)

    def _check_x_data_type(self, X):
        if not isinstance(X, pd.core.frame.DataFrame):
            return pd.DataFrame(X)
        else:
            return X

    def _find_best_feature(self):
        pass

    def _find_best_split(self):
        pass

    def _generate_y_hat(self):
        pass
