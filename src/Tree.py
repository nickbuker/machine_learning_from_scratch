class Node:

    def __init__(self, data=None, depth=0, is_leaf=False):
        """ A general node class for generating tree models

        Parameters
        ----------
        data : Any type
            data saved in node typically:
                - tuple (split col, split value)
                - float with y_hat if leaf
        depth : int
            depth of node (used for tree termination)
        is_leaf : bool
            flags whether or not node is a terminal leaf of the tree
        """
        self.a = None
        self.b = None
        self.data = data
        self.depth = depth
        self.is_leaf = is_leaf

    def query(self, row):
        """ Recursively queries nodes to find y_hat for a row of data

        Parameters
        ----------
        row : numpy array
            row of independent variable data

        Returns
        -------
        float
            y_hat for the row of data
        """
        if self.is_leaf:
            return self.data
        else:
            if row[self.data[0]] > self.data[1]:
                return self.a.query(row)
            else:
                return self.b.query(row)
