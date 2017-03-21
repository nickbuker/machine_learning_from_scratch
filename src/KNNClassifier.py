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

    def predict(self, k, X, weights='distance'):
        """
        Input: int k, numpy array of X test, weights for assignment
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

    def _row_dist(self, row, weights):
        """
        Input: numpy array row of X_test, weights
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
        return self._assign(k_dist, weights)

    def _assign(self, k_dist, weights):
        """
        Input: list containing tuples (distances, y train) and weights
        Output: int assignment for X_test row
        """
        tot_d = sum([n[0] for n in k_dist])
        if weights == 'uniform' or tot_d == 0:
            return int(round(np.mean([n[1] for n in k_dist])))
        if weights == 'distance':
            return int(round(np.mean([1 - (n[0] / tot_d) * n[1] for n in k_dist])))
