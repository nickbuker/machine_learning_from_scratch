import numpy as np
from Tree2 import Node
from scoring import R2, RSS


class DecisionTreeRegressor:
    def __init__(self):
        """
        Decision tree regression implemented in Python using numpy
        """
        pass

    def fit(self, X, y, max_depth):
        """ Takes in training data and generates the decision tree

        Parameters
        ----------
        X : numpy array
            training data independent variable(s)
        y : numpy array
            training data dependent variable
        max_depth : int
            max depth permitted for tree

        Returns
        -------
        None
        """
        self.tree = Node()
        # ensure data is in numpy arrays
        X = self._make_numpy(data=X)
        y = self._make_numpy(data=y)
        self._build_tree(X=X, y=y, max_depth=max_depth, tree=self.tree)

    def predict(self, X):
        """ Estimates y for the test data

        Parameters
        ----------
        X : numpy array
            test data independent variable(s)

        Returns
        -------
        numpy array
            y_hat values for test data
        """
        # ensure data is in numpy arrays
        X = self._make_numpy(data=X)
        # map tree query method across data
        return np.apply_along_axis(func1d=self.tree.query, axis=1, arr=X)

    def score(self, X, y):
        """ Calculates model R squared for the test data

        Parameters
        ----------
        X : numpy array
            test data independent variable(s)
        y : numpy array
            test data dependent variable

        Returns
        -------
        float
            R squared value
        """
        # ensure data is in numpy arrays
        X = self._make_numpy(data=X)
        y = self._make_numpy(data=y)
        y_hat = self.predict(X)
        # calculate R-squared value
        return R2(y, y_hat)

    def _make_numpy(self, data):
        """ Converts input data to numpy array if not isinstance() numpy ndarray

        Parameters
        ----------
        data : structure capable of being converted to numpy array
            input data

        Returns
        -------
        numpy array
            data of the appropriate type
        """
        if not isinstance(data, np.ndarray):
            np.array(data)
        return data

    def _build_tree(self, X, y, max_depth, tree):
        """ Takes in training data and recursively builds the decision tree

        Parameters
        ----------
        X : numpy array
            training data independent variable(s)
        y : numpy array
            training data dependent variable
        max_depth : int
            max depth permitted or tree
        tree : Node class
            layer of decision tree

        Returns
        -------
        None
        """
        # find best split across all columns of data
        col, split, a_mean, b_mean = self._find_best_col(X, y)
        mask = X[:, col] > split
        # update tree with split information
        tree.data = (col, split)
        # Node will be leaf if max_depth reached or contains 3 or less observations
        a_leaf = tree.depth + 1 == max_depth or sum(mask) <= 3
        b_leaf = tree.depth + 1 == max_depth or sum(np.invert(mask)) <= 3
        # create mew nodes
        tree.a = Node(depth=tree.depth + 1, is_leaf=a_leaf)
        tree.b = Node(depth=tree.depth + 1, is_leaf=b_leaf)
        # terminate tree with mean for split or continue to build tree recursively
        if a_leaf:
            tree.a.data = a_mean
        else:
            self._build_tree(X=X[mask],
                             y=y[mask],
                             max_depth=max_depth,
                             tree=tree.a)
        if b_leaf:
            tree.b.data = b_mean
        else:
            self._build_tree(X=X[np.invert(mask)],
                             y=y[np.invert(mask)],
                             max_depth=max_depth,
                             tree=tree.b)

    def _find_best_col(self, X, y):
        """ Iterates through the columns to find the one generating the split producing least error

        Parameters
        ----------
        X : numpy array
            training data independent variable(s)
        y : numpy array
            training data dependent variable

        Returns
        -------
        tuple
            col : int
                index of column giving best split
            split : float or int
                value giving best split
            b_mean : float
                mean of y below or equal to the split value
            a_mean : float
                mean of y above the split value
        """
        # initialize starting values
        error = np.inf
        col = 0
        split = 0
        a_mean = 0
        b_mean = 0
        # for each col, find best split and update values if error improved
        for i in range(0, X.shape[1]):
            temp_error, temp_split, temp_a_mean, temp_b_mean = self._find_best_split(X[:, i], y)
            if temp_error < error:
                error = temp_error
                col = i
                split = temp_split
                a_mean = temp_a_mean
                b_mean = temp_b_mean
        return col, split, a_mean, b_mean

    def _find_best_split(self, values, y):
        """ Iterates through the unique values of the column to find the split producing least error

        Parameters
        ----------
        values : numpy array
            column of dependent variable training data
        y : numpy array
            training data dependent variable

        Returns
        -------
        tuple
            error : float
                sum of squared error
            split : float or int
                value giving best split
            b_mean : float
                mean of y below or equal to the split value
            a_mean : float
                mean of y above the split value
        """
        # initialize starting values
        error = np.inf
        split = 0
        a_mean = 0
        b_mean = 0
        # check all possible splits and update values if error improved
        for n in np.unique(values):
            mask = values > n
            # skip splits resulting in arrays with no values
            if sum(mask) == 0:
                continue
            temp_a_mean = np.mean(y[mask])
            temp_b_mean = np.mean(y[np.invert(mask)])
            y_hat = np.repeat(a_mean, len(y))
            y_hat[np.invert(mask)] = b_mean
            temp_error = RSS(y, y_hat)
            if temp_error < error:
                error = temp_error
                split = n
                a_mean = temp_a_mean
                b_mean = temp_b_mean
        return error, split, a_mean, b_mean
