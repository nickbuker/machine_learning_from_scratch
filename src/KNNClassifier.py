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

    def predict(self, X):
        """
        Input: numpy array of X
        Output: numpy array of y_hat
        """
        pass

    def score(self, y):
        """
        Input: numpy array of y
        Output: TDB
        """
        pass

    def _dist(self, X):
        pass

    def _assign(self, distances):
        pass
