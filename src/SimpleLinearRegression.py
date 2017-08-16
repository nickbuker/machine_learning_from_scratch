import numpy as np
from scoring import R2


class SimpleLinearRegression:

    def __init__(self):
        pass

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
        self.x_train, self.y_train = x, y
        self.x_bar, self.y_bar = np.mean(x), np.mean(y)
        self.b1 = self._find_b1()
        self.b0 = self._find_b0()
        print('y_hat = {} + {} * x'.format(self.b1, self.b0))

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
        self.y_hat = (self.b1 * x) + self.b0
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

    def _find_b1(self):
        """ calculates slope (beta 1) for linear model

        Returns
        -------
        float
            slope (beta 1) for linear model
        """
        return (np.sum((self.x_train - self.x_bar) *
                       (self.y_train - self.y_bar)) /
                np.sum((self.x_train - self.x_bar) ** 2))

    def _find_b0(self):
        """ calculates intercept (beta 0) for linear model

        Returns
        -------
        float
            intercept (beta 0) for linear model
        """
        return self.y_bar - self.b1 * self.x_bar
