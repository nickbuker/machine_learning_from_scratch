import numpy as np
from scoring import accuracy, log_loss


class NeuralNetwork:
    """
    A simple neural network implemented in Python using numpy
    """

    def __init__(self):
        pass

    def fit(self, X, y, in_nodes, hid_nodes, out_nodes, epochs=100000, learning_rate=0.001,
            reg_factor=0.001, random_seed=97, print_loss=False):
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
            number of full passes through training data (default value 100000)
        learning_rate : float
            learning rate for gradient descent (default value 0.001)
        reg_factor : float
            regularization strength (default value 0.01)
        random_seed : int
            optional random seed for initial weights generated
        print_loss : bool
            specifies whether or not to print loss every 10000 epochs during training (default value False)

        Returns
        -------
        None
        """
        # initialize parameters to some random values
        np.random.seed(random_seed)
        nodes = (in_nodes, hid_nodes, out_nodes)
        W1 = np.random.randn(in_nodes, hid_nodes) / np.sqrt(in_nodes)
        b1 = np.zeros((1, hid_nodes))
        W2 = np.random.randn(hid_nodes, out_nodes) / np.sqrt(hid_nodes)
        b2 = np.zeros((1, out_nodes))
        # store model features
        self.model = {'W1': W1, 'b1': b1,
                      'W2': W2, 'b2': b2,
                      'nodes': nodes}
        # use gradient descent to estimate weights and biases
        self._gradient_descent(X, y, epochs, learning_rate, reg_factor, print_loss)

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
        a1, probs = self._forward_propagation(X)
        if prob:
            return probs
        else:
            return np.argmax(probs, axis=1)

    def score(self, X, y, metric='log_loss'):
        """ Calculates the log loss or accuracy of the model

        Parameters
        ----------
        X : numpy array
            test data
        y : numpy array
            actual class labels for test data
        metric : string
            if 'log_loss' then returns log loss
            if 'accuracy' then returns accuracy

        Returns
        -------
        float
            log loss or accuracy depending on the value of the metric parameter
        """
        if metric == 'log_loss':
            proba = self.predict(X, prob=True)
            temp_loss = 0
            # find sum of log_loss for each class
            for i in range(proba.shape[1]):
                # create array of 0 and 1 for each class
                class_i = (y == i).astype(int)
                temp_loss += log_loss(class_i, proba[:,i])
            return temp_loss
        elif metric == 'accuracy':
            preds = self.predict(X, prob=False)
            return accuracy(y, preds)
        else:
            print('valid scoring metrics are log_loss or accuracy')

    def _gradient_descent(self, X, y, epochs, learning_rate, reg_parameter, print_loss):
        """ Optimizes model using gradient descent

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            labels for training data
        epochs : int
            number of full passes through training data
        learning_rate : float
            learning rate for gradient descent (default value 0.001)
        reg_parameter : float
            regularization strength (default value 0.001)
        print_loss : bool
            specifies whether or not to print loss every 100000 epochs during training

        Returns
        -------
        dict
            trained model
        """
        for i in range(epochs):
            a1, probs = self._forward_propagation(X)
            dW1, db1, dW2, db2 = self._back_propagation(X, y, a1, probs)
            # add regularization terms
            dW1 += reg_parameter * self.model['W1']
            dW2 += reg_parameter * self.model['W2']
            # gradient descent parameter update
            self.model['W1'] += -learning_rate * dW1
            self.model['b1'] += -learning_rate * db1
            self.model['W2'] += -learning_rate * dW2
            self.model['b2'] += -learning_rate * db2
            if print_loss and i % 10000 == 0:
                print('Loss after {0} epochs: {1}'.format(i, self._calculate_loss(X, y, reg_parameter)))

    def _forward_propagation(self, X):
        """ Forward propagation method

        Parameters
        ----------
        X : numpy array
            data

        Returns
        -------
        numpy array
            activations
        numpy array
            probabilities of belonging to class
        """
        z1 = np.dot(X, self.model['W1']) + self.model['b1']
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.model['W2']) + self.model['b2']
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return a1, probs

    def _back_propagation(self, X, y, a1, probs):
        """ Back propagation method

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            labels for training data
        a1 : numpy array
            activations
        probs : numpy array
            probabilities of belonging to class

        Returns
        -------
        float
            partial derivative of input layer weights
        float
            partial derivative of input layer biases
        float
            partial derivative of hidden layer weights
        float
            partial derivative of hidden layer biases
        """
        delta3 = probs
        delta3[range(len(a1)), y] -= 1
        dW2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.model['W2'].T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        return dW1, db1, dW2, db2

    def _calculate_loss(self, X, y, reg_parameter):
        """ Calculates total loss on dataset

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            labels for training data
        reg_parameter : float
            regularization strength

        Returns
        -------
        float
            total loss on dataset
        """
        a1, probs = self._forward_propagation(X)
        log_probs = -np.log(probs[range(len(a1)), y])
        loss = np.sum(log_probs)
        loss += (reg_parameter / 2) * (np.sum(np.square(self.model['W1'])) +
                                       np.sum(np.square(self.model['W2'])))
        return (1 / len(a1)) * loss