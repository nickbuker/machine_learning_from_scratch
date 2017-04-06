import numpy as np
from scoring import R2


class MultipleLinearRegression:

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Input: numpy arrays of training data x and y
        Ouput: prints betas for linear model
        """
        self.X = X
        self.y = y
        self.betas = self._find_betas()
        for i, beta in enumerate(self.betas):
            print('B{}: {}'.format(i, beta), end=" ")

    def predict(self, x):
        """
        Input: numpy array of test data x
        Ouput: numpy array of predicted y
        """
        try:
            self.y_hat = x.dot(self.betas)
            return self.y_hat
        except:
            raise AttributeError('Please fit the model before making predictions.')

    def score(self, y):
        """
        Input: numpy array of test data y
        Ouput: R-squared score
        """
        try:
            self.R2 = R2(y, self.y_hat)
            return self.R2
        except:
            raise AttributeError('Please make a prediction before scoring.')

    def _find_betas(self):
        """
        Input: none
        Output: numpy array of betas
        """
        # solve for betas using (X'X)^-1 X'Y
        return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
