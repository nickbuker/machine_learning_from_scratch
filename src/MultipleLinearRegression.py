import numpy as np
from scoring import R2


class MultipleLinearRegression:
    """
    Multiple linear regression implemented in Python using numpy
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """ takes in training data and calculates betas

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            training data

        Returns
        -------
        prints betas for linear model
        """
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        self.betas = self._find_betas(X, y)

    def predict(self, X):
        """ makes predictions on test data

        Parameters
        ----------
        X : numpy array
            test data

        Returns
        -------
        numpy array
            estimated y values for test data
        """
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        y_hat = X.dot(self.betas)
        return y_hat

    def score(self, X, y):
        """ calculates R squared for test data

        Parameters
        ----------
        X : numpy array
            test data
        y : numpy array
            actual y values for test data

        Returns
        -------
        float
            R squared for test data
        """
        y_hat = self.predict(X)
        return R2(y, y_hat)

    def _find_betas(self, X, y):
        """ calculates the beta values for the linear model

        Parameters
        ----------
        X : numpy array
            test data
        y : numpy array
            actual y values for test data

        Returns
        -------
        numpy array
            betas for linear model
        """
        # solve for betas using (X'X)^-1 X'Y
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
