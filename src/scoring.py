import numpy as np


def RSS(y, y_hat):
    """
    Input: numpy arrays of y and y_hat
    Output: residual sum of squares
    """
    return np.sum((y - y_hat) ** 2)


def TSS(y):
    """
    Input: numpy array of y
    Output: total sum of squares
    """
    return np.sum((y - np.mean(y)) ** 2)


def ESS(y, y_hat):
    """
    Input: numpy arrays of y and y_hat
    Output: explained sum of squares
    """
    return np.sum((y_hat - np.mean(y)) ** 2)


def R2(y, y_hat):
    """
    Input: numpy arrays of y and y_hat
    Output: R-squared value
    """
    return 1 - (RSS(y, y_hat) / TSS(y))
