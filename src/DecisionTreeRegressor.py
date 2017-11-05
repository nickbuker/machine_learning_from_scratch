import numpy as np
import pandas as pd
from scoring import R2, RSS
from Tree import Tree


class DecisionTreeRegressor:

    def __init__(self):
        """
        Decision tree regression implemented in Python using numpy and pandas
        """
        pass

    def fit(self, X, y, max_depth):
        """ Takes in training data and generates the decision tree

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            training data for the model
        y : pandas Series or numpy array
            dependent variable training data for the model
        max_depth : int
            max depth at which the tree will be terminated

        Returns
        -------
        None
        """
        self.tree = Tree()
        self.max_depth = max_depth
        X = self._check_data_type_X(X)
        y = self._check_data_type_y(y)
        self._find_best_feature(X=X,
                                y=y,
                                tree=self.tree.tree)

    def predict(self, X):
        """ Estimates y for the test data

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            test data for the model

        Returns
        -------
        numpy array
            estimated y values for the test data
        """
        X = self._check_data_type_X(X)
        return X.apply(self._generate_y_hat, axis=1)

    def score(self, X, y):
        """ Calculates model R squared for the test data

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            test data for the model
        y : pandas Series or numpy array
            dependent variable test data for the model

        Returns
        -------
        float
            R squared value
        """
        X = self._check_data_type_X(X)
        y = self._check_data_type_y(y)
        y_hat = self.predict(X)
        return R2(y, y_hat)

    def _check_data_type_X(self, X):
        """ Checks if X is a pandas DataFrame and if not, converts it to one

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            data

        Returns
        -------
        pandas DataFrame
            data of appropriate type
        """
        if not isinstance(X, pd.core.frame.DataFrame):
            X = pd.DataFrame(X)
        return X

    def _check_data_type_y(self, y):
        """ Checks if y is a pandas Series and if not, converts it to one

        Parameters
        ----------
        y : pandas Series or numpy array
            data

        Returns
        -------
        pandas Series
            data of appropriate type
        """
        if not isinstance(y, pd.core.series.Series):
            y = pd.Series(y)
        return y

    def _find_best_feature(self, X, y, tree, k='root', i=1):
        """ Iterates across all features to find the one allowing the best split
        and updates the tree accordingly

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            training data for the model
        y : pandas Series or numpy array
            dependent variable training data for the model
        tree : decision tree object (nested dicts and lists)
            current layer of the decision tree
        k : str
            key for the current layer of the decision tree
        i : int
            iterator used to terminate regression at max_depth

        Returns
        -------
        None
        """
        results = [np.inf]
        for col in X.columns:
            temp_results = self._find_best_split(col_data=X[col].values,
                                                 y=y.values)
            if temp_results[-1] < results[-1]:
                results = [col] + temp_results  # concat the lists together
        if i == self.max_depth or results[4] == 1 or results[5] == 1:
            tree[k] = [results[0], results[3], {'b': results[1], 'a': results[2]}]
        else:
            tree[k] = [results[0], results[3], {}]
            self._find_best_feature(X=X[X[results[0]] <= results[3]],
                                    y=y[X[results[0]] <= results[3]],
                                    tree=tree[k][-1],
                                    k='b',
                                    i=i + 1)
            self._find_best_feature(X=X[X[results[0]] > results[3]],
                                    y=y[X[results[0]] > results[3]],
                                    tree=tree[k][-1],
                                    k='a',
                                    i=i + 1)

    def _find_best_split(self, col_data, y):
        """ Iterates across all values of a features to find the best split

        Parameters
        ----------
        col_data : numpy array
            independent variable training data for the the given feature
        y : numpy array
            dependent variable training data

        Returns
        -------
        list
            [0] = y_hat below or equal to the split
            [1] = y_hat above the split
            [2] = split value
            [3] = count of values below or equal to the split
            [4] = count of values above the split
        """
        results = [np.inf]
        vals = set(col_data)
        for val in vals:
            mask_b = col_data <= val
            mean_b = np.mean(y[mask_b])
            mean_a = np.mean(y[col_data > val])
            if sum(mask_b) == 0 or len(y) - sum(mask_b) == 0:
                continue
            else:
                y_hat = np.repeat(mean_a, len(y))
                y_hat[mask_b] = mean_b
                temp_score = RSS(y, y_hat)
                if temp_score < results[-1]:
                    results = [mean_b, mean_a, val, sum(mask_b), len(y) - sum(mask_b), temp_score]
        return results

    def _generate_y_hat(self, row, tree=None):
        """ Estimates y_hat for each row of the test data by recursively querying
        the decision tree

        Parameters
        ----------
        row : pandas Series
            row of test dataset
        tree : decision tree object (nested dicts and lists)
            current layer of the decision tree

        Returns
        -------
        float
            y_hat for the given row of the dataset
        """
        if tree is None:
            tree = self.tree.tree['root']
        if row[tree[0]] <= tree[1]:
            if isinstance(tree[-1]['b'], list):
                return self._generate_y_hat(row=row,
                                            tree=tree[-1]['b'])
            else:
                return tree[-1]['b']
        else:
            if isinstance(tree[-1]['a'], list):
                return self._generate_y_hat(row=row,
                                            tree=tree[-1]['a'])
            else:
                return tree[-1]['a']
