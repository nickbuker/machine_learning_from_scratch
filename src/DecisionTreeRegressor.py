# TODO data type checker for y

import numpy as np
import pandas as pd
from scoring import R2, RSS
from Tree import Tree


class DecisionTreeRegressor:

    def __init__(self):
        pass

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
        self.tree = Tree()
        self.max_depth = max_depth
        X = self._check_data_type_X(X)
        y = self._check_data_type_y(y)
        self._find_best_feature(X=X,
                                y=y,
                                tree=self.tree.tree)

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        X = self._check_data_type_X(X)
        return X.apply(self._generate_y_hat, axis=1)



    def score(self, y, y_hat):
        """

        Parameters
        ----------
        y
        y_hat

        Returns
        -------

        """
        return R2(y, y_hat)

    def _check_data_type_X(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        if not isinstance(X, pd.core.frame.DataFrame):
            X = pd.DataFrame(X)
        return X

    def _check_data_type_y(self, y):
        """

        Parameters
        ----------
        y

        Returns
        -------

        """
        if not isinstance(y, pd.core.series.Series):
            y = pd.Series(y)
        return y

    def _find_best_feature(self, X, y, tree, k='root', i=1):
        """

        Parameters
        ----------
        X
        y
        tree
        k
        i

        Returns
        -------

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
        """

        Parameters
        ----------
        col_data
        y

        Returns
        -------

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
        """

        Parameters
        ----------
        tree

        Returns
        -------

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
