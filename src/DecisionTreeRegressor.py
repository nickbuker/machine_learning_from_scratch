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
        data = self._check_x_data_type(X)
        data.loc[:, 'y'] = y
        data_cols = [col for col in data.columns if col != 'y']
        self._find_best_feature(data=data,
                                data_cols=data_cols,
                                tree=self.tree.tree)

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        X = self._check_x_data_type(X)
        return X.apply(self._generate_y_hat)



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

    def _check_x_data_type(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        if not isinstance(X, pd.core.frame.DataFrame):
            return pd.DataFrame(X)
        else:
            return X

    def _find_best_feature(self, data, data_cols, tree, k='root', i=1):
        """

        Parameters
        ----------
        data
        data_cols
        tree
        k
        i

        Returns
        -------

        """
        results = [np.inf]
        for col in data_cols:
            temp_results = self._find_best_split(col=data[col].values,
                                                 y=data['y'].values)
            if temp_results[-1] < results[-1]:
                results = [col] + temp_results  # concat the lists together
        if i == self.max_depth or results[4] == 1 or results[5] == 1:
            tree[k] = [results[0], results[3], {'b': results[1], 'a': results[2]}]
        else:
            tree[k] = [results[0], results[3], {}]
            self._find_best_feature(data=data[data[results[0]] <= results[3]],
                                    data_cols=data_cols,
                                    tree=tree[k][-1],
                                    k='b',
                                    i=i + 1)
            self._find_best_feature(data=data[data[results[0]] > results[3]],
                                    data_cols=data_cols,
                                    tree=tree[k][-1],
                                    k='a',
                                    i=i + 1)

    def _find_best_split(self, col, y):
        """

        Parameters
        ----------
        col
        y

        Returns
        -------

        """
        results = [np.inf]
        vals = set(col)
        for val in vals:
            mask_b = col <= val
            mean_b = np.mean(y[mask_b])
            mean_a = np.mean(y[col > val])
            if sum(mask_b) == 0 or len(y) - sum(mask_b) == 0:
                continue
            else:
                y_hat = np.repeat(mean_a, len(y))
                y_hat[mask_b] = mean_b
                temp_score = RSS(y, y_hat)
                if temp_score < results[-1]:
                    results = [mean_b, mean_a, val, sum(mask_b), len(y) - sum(mask_b), temp_score]
        return results

    def _generate_y_hat(self, row, tree=self.tree.tree['root']):
        """

        Parameters
        ----------
        row
        tree

        Returns
        -------

        """
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
        