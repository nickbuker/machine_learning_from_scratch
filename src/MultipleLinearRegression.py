import numpy as np


class MultipleLinearRegression(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.betas = self._matix()
        for i, beta in enumerate(self.betas):
            print 'B{}:  {}'.format(i, beta)

    def predict(self):
        pass

    def _matrix(self):
        # (X'X)^-1 X'Y
        return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
