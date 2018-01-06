import numpy as np
from scoring import R2


class SimpleLinearRegression:
    """
    Simple linear regression implemented in Python using numpy
    """

    def __init__(self):
        self.model = {}

    def fit(self, x, y):
        """ takes in training data and calculates betas

        Parameters
        ----------
        x : numpy array
            training data
        y : numpy array
            training data

        Returns
        -------
        prints linear equation for trained model
        """
        x_bar, y_bar = np.mean(x), np.mean(y)
        self._find_b1(x=x, x_bar=x_bar, y=y, y_bar=y_bar)
        self._find_b0(x_bar=x_bar, y_bar=y_bar)

    def predict(self, x):
        """ makes predictions on test data

        Parameters
        ----------
        x : numpy array
            test data

        Returns
        -------
        numpy array
            estimated y values for test data
        """
        y_hat = (self.model['b1'] * x) + self.model['b0']
        return y_hat

    def score(self, x, y):
        """ calculates R squared for test data

        Parameters
        ----------
        x : numpy array
            test data
        y : numpy array
            actual y values for test data

        Returns
        -------
        float
            R squared for test data
        """
        y_hat = (self.model['b1'] * x) + self.model['b0']
        return R2(y, y_hat)

    def _find_b1(self, x, x_bar, y, y_bar):
        """ calculates slope (beta 1) for linear model and adds it to model dict

        Parameters
        ----------
        x : numpy array
            training data
        x_bar : float
            mean of training data
        y : numpy array
            actual y values for training data
        y_bar : float
            mean of y values for training data

        Returns
        -------
        None
        """
        self.model['b1'] = (np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2))

    def _find_b0(self, x_bar, y_bar):
        """ calculates intercept (beta 0) for linear model and adds it to model dict

        Parameters
        ----------
        x_bar : float
            mean of y values for training data
        y_bar : float
            mean of training data

        Returns
        -------
        None
        """
        self.model['b0'] = y_bar - self.model['b1'] * x_bar
