# Machine Learning From Scratch

##### Simple machine learning algorithms implemented from scratch in Python for the purposes of fun and education (use at your own risk). Under construction: More methods to come.

```Python
import numpy as np
from scoring import R2


class SimpleLinearRegression(object):

    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Input: numpy arrays of training data x and y
        Ouput: prints linear equation for trained model
        """
        self.x_train, self.y_train = x, y
        self.x_bar, self.y_bar = np.mean(x), np.mean(y)
        self.b1 = self._find_b1()
        self.b0 = self._find_b0()
        print 'y_hat = {} + {} * x'.format(self.b1, self.b0)

    def predict(self, x):
        """
        Input: numpy array of test data x
        Ouput: numpy array of predicted y
        """
        try:
            self.y_hat = (self.b1 * x) + self.b0
            return self.y_hat
        except AttributeError:
            print 'Please fit the model before making predictions.'

    def score(self, y):
        """
        Input: numpy array of test data y
        Ouput: R-squared score
        """
        try:
            self.R2 = R2(y, self.y_hat)
            return self.R2
        except AttributeError:
            print 'Please make a prediction before scoring.'

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
```
