import numpy as np
from Node import Node
from scoring import R2, RSS

class DecisionTreeRegressor:
    def __init__(self):
        self.tree = Node()

    def fit(self, X, y, max_depth):
        """

        Parameters
        ----------
        X
        y
        max_depth

        Returns
        -------

        """
        X = self._make_numpy(data=X)
        y = self._make_numpy(data=y)
        self._build_tree(X=X, y=y, max_depth=max_depth, tree=self.tree)

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        X =  self._make_numpy(data=X)
        return np.apply_along_axis(func1d=self._query_tree, axis=1, arr=X)

    def score(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        X = self._make_numpy(data=X)
        y = self._make_numpy(data=y)
        y_hat = self.predict(X)
        return R2(y, y_hat)

    def _make_numpy(self, data):
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        if not isinstance(data, np.ndarray):
            np.array(data)
        return data

    def _build_tree(self, X, y, max_depth, tree):
        """

        Parameters
        ----------
        X
        y
        max_depth
        tree

        Returns
        -------

        """
        col, split, b_mean, a_mean = self._find_best_col(X, y)
        mask = X[:, col] <= split
        tree.data = (col, split)
        is_leaf = tree.depth + 1 == max_depth
        tree.add_child(key='b', depth=tree.depth + 1, is_leaf=is_leaf)
        tree.add_child(key='a', depth=tree.depth + 1, is_leaf=is_leaf)
        if is_leaf:
            tree.children['b'].data = b_mean
            tree.children['a'].data = a_mean
        else:
            self._build_tree(X=X[mask],
                             y=y[mask],
                             max_depth=max_depth,
                             tree=tree.children['b'])
            self._build_tree(X=X[np.invert(mask)],
                             y=y[np.invert(mask)],
                             max_depth=max_depth,
                             tree=tree.children['a'])
    # TODO continue implementation (min leaf size)

    def _find_best_col(self, X, y):
        """ Iterates through the columns to find the one generating the split producing
        least error

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        error = np.inf
        col = 0
        split = 0
        b_mean = 0
        a_mean = 0
        for i in range(0, X.shape[1]):
            temp_error, temp_split, temp_b_mean, temp_a_mean = self._find_best_split(X[:, i], y)
            if temp_error < error:
                error = temp_error
                col = i
                split = temp_split
                b_mean = temp_b_mean
                a_mean = temp_a_mean
        return col, split, b_mean, a_mean

    def _find_best_split(self, col_values, y):
        """ Iterates through the unique values of the column to find the split producing
        least error

        Parameters
        ----------
        col_values
        y

        Returns
        -------

        """
        error = np.inf
        split = 0
        b_mean = 0
        a_mean = 0
        for n in np.unique(col_values):
            mask = col_values <= n
            temp_b_mean = np.mean(y[mask])
            temp_a_mean = np.mean(y[np.invert(mask)])
            y_hat = np.array(sum(mask) * [temp_b_mean] + sum(np.invert(mask)) * [temp_a_mean])
            temp_error = RSS(y, y_hat)
            if temp_error < error:
                error = temp_error
                split = n
                b_mean = temp_b_mean
                a_mean = temp_a_mean
        return error, split, b_mean, a_mean

    def _query_tree(self, row):
        """ Takes in row of data and queries tree to find y_hat

        Parameters
        ----------
        row : numpy array
            row of data from X

        Returns
        -------
        float
            y_hat for particular row
        """
        return self.tree.get_leaf(row=row)