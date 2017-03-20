import numpy as np


class KNNClassifier(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Input: numpy arrays of X and y
        Output: none
        """
        self.X_train, self.y_train = X, y

    def predict(self, k, X):
        """
        Input: numpy array of X test
        Output: numpy array of y_hat
        """
        self.k = k
        return np.apply_along_axis(self._row_dist, 1, X)

    def score(self, y):
        """
        Input: numpy array of y
        Output: TDB
        """
        pass

    def _row_dist(self, row):
        """
        Input: numpy array row of X_test
        Output: integer y_hat for row
        """
        k_dist = []
        for i, r in enumerate(self.X_train):
            d = np.linalg.norm(row - r)
            if len(k_dist) < self.k:
                k_dist.append((d, self.y_train[i]))
                k_dist.sort()
            elif d < k_dist[-1][0]:
                k_dist[-1] = (d, self.y_train[i])
                k_dist.sort()
        return self._assign(k_dist)

    def _assign(self, k_dist):
        """
        Input: list containing tuples with distances and y train
        Output: int assignment for X_test row
        """
        return int(round(np.mean([n[1] for n in k_dist])))
