import numpy as np
from scoring import accuracy


class KNNClassifier:

    def __init__(self):
        pass

    def fit(self, X, y):
        """ Takes in training data

        Parameters
        ----------
        X : numpy array
            training data
        y :numpy array
            actual 0 or 1 class labels for training data

        Returns
        -------
        None
        """
        self.X_train, self.y_train = X, y

    def predict(self, k, X, weights='distance'):
        """ Make class predictions

        Parameters
        ----------
        k : int
            number of nearby points to consider in classification
        X : numpy array
            test data
        weights : basestring
            uniform considers k points evenly and distance weighs nearby points more heavily

        Returns
        -------
        numpy array
            class predictions made by model
        """
        try:
            self.k = k
            self.y_pred = np.apply_along_axis(self._row_dist, 1, X)
            return self.y_pred
        except:
            raise AttributeError('Please fit the model before making predictions.')

    def score(self, y):
        """ Calculates the accuracy of the model

        Parameters
        ----------
        y : numpy array
            actual labels for data

        Returns
        -------
        float
            accuracy of model
        """
        try:
            self.accuracy = accuracy(y, self.y_pred)
            return self.y_pred
        except:
            raise AttributeError('Please fit the model before making predictions.')

    def _row_dist(self, row, weights):
        """ Calculate distances to each point in the training data

        Parameters
        ----------
        row : int
            row number in X_test
        weights : basestring
            uniform considers k points evenly and distance weighs nearby points more heavily

        Returns
        -------
        int
            class assignment for test data row
        """
        k_dist = []
        if weights == 'uniform' or weights == 'distance':
            for i, r in enumerate(self.X_train):
                d = np.linalg.norm(row - r)
                if len(k_dist) < self.k:
                    k_dist.append((d, self.y_train[i]))
                    k_dist.sort()
                elif d < k_dist[-1][0]:
                    k_dist[-1] = (d, self.y_train[i])
                    k_dist.sort()
            return self._assign(k_dist, weights)
        else:
            raise ValueError('Weight types: uniform or distance')

    def _assign(self, k_dist, weights):
        """ Assigns label to point

        Parameters
        ----------
        k_dist : list of tuples
            contains distance and label for training data
        weights : basestring
            uniform considers k points evenly and distance weighs nearby points more heavily

        Returns
        -------
        int
            class assignment for test data row
        """
        tot_d = sum([n[0] for n in k_dist])
        if weights == 'uniform' or tot_d == 0:
            return int(round(np.mean([n[1] for n in k_dist])))
        if weights == 'distance':
            return int(round(np.mean([1 - (n[0] / tot_d) * n[1] for n in k_dist])))
