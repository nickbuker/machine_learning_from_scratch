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
        X : numpy array
            training data
        y : numpy array
            actual 0 and 1 class labels for training data
        intercept : bool
            if True include intercept
            if False do not include intercept
        learning_rate : float
            learning rate for gradient descent
        converge_change : float
            threshold for reaching convergence during gradient descent
        max_iter : int
            maximum number of iterations permitted during gradient descent
            (prevents infinite loop if convergence not reached)

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

    def _gradient_descent(self, X, y, intercept, learning_rate, convergence_change, max_iter):
        """ Solves for betas using gradient descent

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            actual 0 and 1 class labels for test data
        intercept : bool
            if True include intercept
            if False do not include intercept
        learning_rate : float
            learning rate for gradient descent
        converge_change : float
            threshold for reaching convergence during gradient descent
        max_iter : int
            maximum number of iterations permitted during gradient descent
            (prevents infinite loop if convergence not reached)

        Returns
        -------
        numpy array
            beta values for logistic regression model
        """
        if intercept:
            print('intercept not yet implemented')
            return
        else:
            betas = np.zeros(X.shape[1])
            # TODO continue writing me

    def _logit(self, X, betas):
        return 1 / (1 + np.exp(-X.dot(betas)))
