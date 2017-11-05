import numpy as np
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
        rows = [self._sample_rows(X.index) for _ in range(n_estimators)]
        n_cols = int(len(X.columns) ** 0.5)
        cols = [self._sample_cols(X.columns, n_cols) for _ in range(n_estimators)]
        for i, tree in enumerate(self.trees):
            tree.fit(X[cols[i]].iloc[rows[i]], y[rows[i]])

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
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
        y_hat = self.predict(X)
        return R2(y, y_hat)

    def _sample_rows(self, index):
        """

        Parameters
        ----------
        index

        Returns
        -------

        """
        return np.random.randint(0, len(index), len(index))

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

