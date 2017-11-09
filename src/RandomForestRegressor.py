import numpy as np
import pandas as pd
from scoring import R2, RSS
from DecisionTreeRegressor import DecisionTreeRegressor


class RandomForestRegressor:

    def __init__(self):
        """

        """
        pass

    def fit(self, X, y, max_depth, n_estimators):
        """

        Parameters
        ----------
        X
        y
        max_depth
        n_estimators

        Returns
        -------

        """
        # instantiate forest of decision trees
        self.trees = [DecisionTreeRegressor() for _ in range(n_estimators)]
        # create masks for rows and columns of data for each tree
        X = self._check_data_type_X(X)
        y = self._check_data_type_y(y)
        rows = [self._sample_rows(X.index) for _ in range(n_estimators)]
        n_cols = int(len(X.columns) ** 0.5)
        cols = [self._sample_cols(X.columns, n_cols) for _ in range(n_estimators)]
        for i, tree in enumerate(self.trees):
            # TODO fix this line
            tree.fit(X[cols[i]].iloc[rows[i]], y.iloc[rows[i]], max_depth)

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        X = self._check_data_type_X(X)
        preds = []
        for tree in self.trees:
            preds.append(tree.predict(X))
        return np.mean(preds, axis=0)

    def score(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

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
        return X.reset_index(drop=True, inplace=False)

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
        return y.reset_index(drop=True, inplace=False)

    def _sample_rows(self, index):
        """

        Parameters
        ----------
        index

        Returns
        -------

        """
        return np.random.randint(0, len(index) -1 , len(index))

    def _sample_cols(self, cols, n_cols):
        """

        Parameters
        ----------
        cols
        n_cols

        Returns
        -------

        """
        return np.random.choice(cols, n_cols)
