import numpy as np


def RSS(y, y_hat):
    """ Residual Sum of Squares

    :param y: numpy array
        y values
    :param y_hat: numpy array
        y values estimated by model

    :return: float
        residual sum of squares
    """
    return np.sum((y - y_hat) ** 2)


def TSS(y):
    """ Total Sum of Squares

    :param y: numpy array
        y values

    :return: float
        total sum of squares
    """
    return np.sum((y - np.mean(y)) ** 2)


def ESS(y, y_hat):
    """ Explained Sum of Squares

    :param y: numpy array
        y values
    :param y_hat: numpy array
        yes values estimated by model

    :return: float
        explained sum of squares
    """
    return np.sum((y_hat - np.mean(y)) ** 2)


def R2(y, y_hat):
    """ R squared (coefficient of determination)

    :param y: numpy array
        y values
    :param y_hat: numpy array
        y values estimated by model

    :return: float
        R squared
    """
    return 1 - (RSS(y, y_hat) / TSS(y))


def accuracy(y, y_pred):
    """ Accuracy of classification model

    :param y: numpy array
        class labels
    :param y_pred: numpy array
        classes predicted by model

    :return: float
        accuracy
    """
    return sum(y == y_pred) / len(y)


def log_loss(y, y_prob):
    """ Log Loss

    :param y: numpy array
        0 or 1 class labels
    :param y_prob: numpy array
        estimated probability of belonging to class 1

    :return: float
        log loss
    """
    return -sum((y * np.log(y_prob)) + ((1 - y) * np.log(1 - y_prob))) / len(y)
