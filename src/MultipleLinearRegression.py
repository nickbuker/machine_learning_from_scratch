import numpy as np
from scoring import R2


class MultipleLinearRegression:

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
        self.X_train = np.insert(X, 0, intercept_col, axis=1)
        self.y_train = y
        self.betas = self._find_betas()
        for i, beta in enumerate(self.betas):
            print('B{}: {}'.format(i, beta), end=" ")

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
        self.X_test = np.insert(X, 0, intercept_col, axis=1)
        self.y_hat = self.X_test.dot(self.betas)
        return self.y_hat

    def score(self, y):
        """ calculates R squared for test data

        Parameters
        ----------
        y : numpy array
            actual y values for test data

        Returns
        -------
        float
            R squared for test data
        """
        self.R2 = R2(y, self.y_hat)
        return self.R2

    def _find_betas(self):
        """ calculates the beta values for the linear model

        Returns
        -------
        numpy array
            betas for linear model
        """
        # solve for betas using (X'X)^-1 X'Y
        return np.linalg.inv(self.X_train.T.dot(self.X_train)).dot(self.X_train.T).dot(self.y_train)
