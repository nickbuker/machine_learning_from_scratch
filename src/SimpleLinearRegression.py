import numpy as np


class SimpleLinearRegression(object):

    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Input: numpy arrays x and y
        Ouput: prints linear equation for trained model
        """
        self.x_train, self.y_train = x, y
        self.x_bar, self.y_bar = np.mean(x), np.mean(y)
        self.b1 = self._find_b1()
        self.b0 = self._find_b0()
        self.RSS = self._find_RSS(train=True)
        print 'y_hat = {} + {} * x'.format(self.b1, self.b0)
        print 'RSS = {}'.format(self.RSS)

    def _find_b1(self):
        """
        Input: none
        Output: slope for linear model
        """
        return (np.sum((self.x_train - self.x_bar) *
                       (self.y_train - self.y_bar)) /
                np.sum((self.x_train - self.x_bar) ** 2))

    def _find_b0(self):
        """
        Input: none
        Output: intercept for linear model
        """
        return self.y_bar - self.b1 * self.x_bar

    def _find_RSS(self, train=True):
        """
        Input: none
        Output: residual sum of squares
        """
        if train:
            y_hat = (self.b1 * self.x_train) + self.b0
            return sum((self.y_train - y_hat) ** 2)
