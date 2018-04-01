import numpy as np


def RSS(y, y_hat):
    """ Residual Sum of Squares

    Parameters
    ----------
    y : numpy array
        y values
    y_hat : numpy array
        y values estimated by model

    Returns
    -------
    float
        residual sum of squares
    """
    return np.sum((y - y_hat) ** 2)


def TSS(y):
    """ Total Sum of Squares

    Parameters
    ----------
    y : numpy array
        y values

    Returns
    -------
    float
        total sum of squares
    """
    return np.sum((y - np.mean(y)) ** 2)


def ESS(y, y_hat):
    """ Explained Sum of Squares

    Parameters
    ----------
    y : numpy array
        y values
    y_hat : numpy array
        y values estimated by model

    Returns
    -------
    float
        explained sum of squares
    """
    return np.sum((y_hat - np.mean(y)) ** 2)


def R2(y, y_hat):
    """ R squared (coefficient of determination)

    Parameters
    ----------
    y : numpy array
        y values
    y_hat : numpy array
        y values estimated by model

    Returns
    -------
    float
        R squared
    """
    return 1 - (RSS(y, y_hat) / TSS(y))


def accuracy(y, y_pred):
    """ Accuracy of classification model

    Parameters
    ----------
    y : numpy array
        class labels
    y_pred : numpy array
        classes predicted by model

    Returns
    -------
    float
        accuracy
    """
    return sum(y == y_pred) / y.shape[0]


def log_loss(y, y_prob, padding=True):
    """ Log Loss

    Parameters
    ----------
    y : numpy array
        actual 0 or 1 class labels
    y_prob : numpy array
        estimated probability of belonging to class 1
    padding : bool
        whether or not to pad zeros and ones with 1*10^-15

    Returns
    -------
    float
        log loss
    """
    if padding:
        y_prob[y_prob < np.power(10.0, -15)] = np.power(10.0, -15)
        y_prob[y_prob > (np.float64(1.0) - np.power(10.0, -15))] = np.float64(1.0) - np.power(10.0, -15)
    return -sum((y * np.log(y_prob)) + ((np.float64(1.0) - y) * np.log(np.float64(1.0) - y_prob))) / y.shape[0]
