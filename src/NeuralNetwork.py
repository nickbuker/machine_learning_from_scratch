from LogisticRegression import LogisticRegression


class NeuralNetwork:

    def __init__(self):
        pass

    def fit(self, X, y, in_out_nodes, hidden_nodes):
        """ Takes in test data and trains model

        Parameters
        ----------
        X : numpy array
            training data
        y : numpy array
            labels for training data
        in_out_nodes : int
            number of nodes in input and output layers
        hidden_nodes
            number of nodes in hidden layer
        Returns
        -------
        None
        """
        pass

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