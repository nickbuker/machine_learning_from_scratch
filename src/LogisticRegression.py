from scoring import log_loss, accuracy
import numpy as np


class LogisticRegression:

    def __init__(self):
        pass

    def fit(self, X, y, learning_rate=0.001,
            converge_change=0.001, max_iter=10000):
        """ Takes in training data and calculates betas

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            actual 0 and 1 class labels for training data
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
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        self.betas = self._gradient_descent(X=X,
                                            y=y,
                                            learning_rate=learning_rate,
                                            convergence_change=converge_change,
                                            max_iter=max_iter)

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
            probabilities or class predictions for test data depending on the prob parameter
        """
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        self.y_prob = self._logit(X, self.betas)
        if prob:
            return self.y_prob
        else:
            self.y_pred = np.zeros(len(self.y_prob))
            # if prob in y_prob >= threshold, convert label to 1
            np.place(self.y_pred, self.y_prob >= threshold, 1)
            return self.y_pred

    def score(self, y, metric='log_loss'):
        """ Scores the predictions

        Parameters
        ----------
        y : numpy array
            actual 0 and 1 class labels for test data
        metric : string
            if 'log_loss' then returns log loss
            if 'accuracy' then returns accuracy

        Returns
        -------
        float
            log loss or accuracy score depending on the metric parameter
        """
        if metric == 'log_loss':
            return log_loss(y, self.y_prob)
        elif metric == 'accuracy':
            return accuracy(y, self.y_pred)
        else:
            print('valid scoring metrics are log_loss or accuracy')

    def _gradient_descent(self, X, y, learning_rate, convergence_change, max_iter):
        """ Estimates betas using gradient descent

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            actual 0 and 1 class labels for test data
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
        # initialize variables
        betas = np.zeros(X.shape[1])
        y_prob = self._logit(X, betas)
        loss = log_loss(y, y_prob)
        change = 1
        i = 1
        # loop until convergence or max iterations are reached
        while change > convergence_change and i <= max_iter:
            old_loss = loss
            betas = betas - (learning_rate * self._gradient(betas, X, y))
            y_prob = self._logit(X, betas)
            loss = log_loss(y, y_prob)
            change = old_loss - loss
            i += 1
        if i == max_iter:
            print('failed to reach convergence')
        return betas

    def _logit(self, X, betas):
        """

        Parameters
        ----------
        X : numpy array
            data
        betas : numpy array
            beta values for linear equation
        Returns
        -------
        numpy array
            probabilities of belonging to class 1
        """
        return 1 / (1 + np.exp(-X.dot(betas)))

    def _gradient(self, betas, X, y):
        """ Calculates the gradient

        Parameters
        ----------
        betas
        X : numpy array
            training data
        y : : numpy array
            actual 0 and 1 class labels for test data

        Returns
        -------
        numpy array
            gradient
        """
        diffs = self._logit(X, betas) - y
        return diffs.T.dot(X)
