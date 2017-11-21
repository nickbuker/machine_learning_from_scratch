import numpy as np
from scoring import R2
from DecisionTreeRegressor import DecisionTreeRegressor


class RandomForestRegressor:

    def __init__(self):
        """
        Decision tree regression implemented in Python using numpy
        """
        pass

    def fit(self, X, y, max_depth, n_estimators):
        """ Takes in training data and generates the random forest

        Parameters
        ----------
        X : numpy array
            training data independent variable(s)
        y : numpy array
            training data dependent variable
        max_depth : int
            max depth permitted for tree
        n_estimators : int
            number of tree for forest

        Returns
        -------
        None
        """
        # instantiate forest of decision trees
        self.trees = [DecisionTreeRegressor() for _ in range(n_estimators)]
        # ensure data is in numpy arrays
        X = self._make_array(X)
        y = self._make_array(y)
        # sample rows and columns for each tree
        rows = [self._draw_sample(X.shape[0], X.shape[0], replacement=True)
                for _ in range(n_estimators)]
        n_cols = int(X.shape[1] ** 0.5)
        cols = [self._draw_sample(X.shape[1], n_cols, replacement=False)
                for _ in range(n_estimators)]
        col_map = [self._make_col_map(idxs) for idxs in cols]
        # fit trees to these samples
        for i, tree in enumerate(self.trees):
            tree.fit(X=X[rows[i], :][:, cols[i]],
                     y=y[rows[i]],
                     max_depth=max_depth,
                     col_map=col_map[i])

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
        X = self._make_array(X)
        preds = []
        # take the mean of the estimates of all trees for each data point
        for tree in self.trees:
            preds.append(tree.predict(X))
        return np.mean(preds, axis=0)

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
            R-squared value
        """
        # ensure data is in numpy arrays
        X = self._make_array(X)
        y = self._make_array(y)
        y_hat = self.predict(X)
        # calculate R-squared value
        return R2(y, y_hat)

    def _make_array(self, data):
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

    def _draw_sample(self, sample_max, sample_size, replacement):
        """ Takes in information about max value and sample size and generates
        a numpy array of ints used for sampling rows and columns for forest

        Parameters
        ----------
        sample_max : int
            the max value (exclusive) to be contained in sample index array
        sample_size : int
            number of elements sample index array should contain
        replacement : bool
            flag indicating whether or not there should be replacement

        Returns
        -------
        numpy array
            array of ints to serve as sampling index for rows or columns
        """
        # create an array of integers (with replacement) as index for sampling
        if replacement:
            return np.random.randint(0, sample_max, sample_size)
        # create an array of integers (w/o replacement) as index for sampling
        else:
            temp_array = np.arange(sample_max)
            np.random.shuffle(temp_array)
            return temp_array[0: sample_size]

    def _make_col_map(self, idxs):
        # TODO finish docstring
        """

        Parameters
        ----------
        idxs : numpy array
            indexes of

        Returns
        -------
        dict
            mapping of new column indices to old column indices
        """
        return dict(zip(range(idxs.shape[0]), idxs))
