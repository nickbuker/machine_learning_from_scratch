import numpy as np


class SimpleLinearRegression(object):

    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Input: numpy arrays x and y
        Ouput: prints linear equation for trained model
        """
        self.x, self.y = x, y
        self.x_bar, self.y_bar = np.mean(x), np.mean(y)
        self.b1 = self._find_b1()
        self.b0 = self._find_b0()
        print 'y_hat = {} + {} * x'.format(self.b1, self.b0)

    def _find_b1(self):
        """
        Input: none
        Output: slope for linear model
        """
        return (np.sum((self.x - self.x_bar) * (self.y - self.y_bar)) /
                np.sum((self.x - self.x_bar) ** 2))

    def _find_b0(self):
        """
        Input: none
        Output: intercept for linear model
        """
        return self.y_bar - self.b1 * self.x_bar

    def _find_RSS(self, y_test=None):
        """
        Input: none
        Output: residual sum of squares
        """
