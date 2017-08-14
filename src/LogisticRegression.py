from scoring import log_loss, accuracy
import numpy as np


class LogisticRegression:

    def __init__(self):
        pass

    def fit(self, X, y, intercept=False, learning_rate=0.001,
            converge_change=0.001, max_iter=10000):
        """ Takes in training data and calculates betas

        Parameters
        ----------
        X
        y
        intercept
        learning_rate
        converge_change : float
            threshold for reaching convergence
        max_iter : int
            maximum number of iterations permitted (prevent infinite loops if
            convergence not reached)

        Returns
        -------
        None
        """
        pass

    def predict(self, X, prob=True, threshold=0.5):
        """ Makes probability or class predictions for test data

        Parameters
        ----------
        X : numpy array
           test data
        prob : bool
            if True, returns probability of class 1
            if False, returns class prediction
        threshold : float
            if prob is False, sets probability threshold for class 1
            if prob is True, this argument has no effect

        Returns
        -------
        numpy array
            probabilities or predictions for test data depending on the prob parameter
        """
        pass

    def score(self, y, metric='log_loss'):
        """

        Parameters
        ----------
        y : numpy array
            actual 0 and 1 class labels for test data
        metric : string


        Returns
        -------
        float
            log loss or accuracy score depending on the metric parameter
        """
        pass