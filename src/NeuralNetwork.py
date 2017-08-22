import numpy as np


class NeuralNetwork:

    def __init__(self):
        pass

    def fit(self, X, y, in_nodes, hid_nodes, out_nodes, epochs=10000,
            learning_rate=0.01, reg_factor=0.01):
        """ Takes in test data and trains model

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            labels for training data
        in_nodes : int
            number of nodes in the input layer
        hid_nodes : int
            number of nodes in hidden layer
        out_nodes : int
            number of nodes in output layer
        epochs : int
            number of full passes through training data (default value 10000)
        learning_rate : float
            learning rate for gradient descent (default value 0.01)
        reg_factor : float
            regularization strength (default value 0.01)

        Returns
        -------
        None
        """
        # initialize parameters to some random values
        weights1 = np.random.randn(in_nodes, hid_nodes) / np.sqrt(in_nodes)
        betas1 = np.zeros((1, hid_nodes))
        weights2 = np.random.randn(hid_nodes, out_nodes) / np.sqrt(hid_nodes)
        betas2 = np.zeros(1, out_nodes)
        model = {'weights1': weights1, 'betas1': betas1,
                 'weights2': weights2, 'betas2': betas2}

    def predict(self, X, prob=True):
        """ Makes probability or class predictions for test data

        Parameters
        ----------
        X : numpy array
            test data
        prob : bool
            if True, returns probabilities for each class
            if False, returns class prediction

        Returns
        -------
        numpy array
            probabilities or class predictions for test data depending on the prob parameter
        """
        pass

    def _gradient_descent(self, learning_rate, reg_parameter):
        pass

    def _forward_propagation(self):
        pass

    def _loss_function(self):
        pass

    def _back_propagation(self):
        pass