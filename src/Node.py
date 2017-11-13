class Node:
    def __init__(self, data=None, is_leaf=False):
        """ Node class for implementation of machine learning tree models

        Parameters
        ----------
        data : tuple or float
            if not is_leaf then tuple (split_col, split_val)
            if is_leaf then float value of y_hat
        is_leaf : bool
            whether or not this node is a terminal leaf
        """
        self.data = data
        self.is_leaf = is_leaf
        self.children = {}

    def add_child(self, key, data, is_leaf):
        """ Adds child Node to self.children

        Parameters
        ----------
        key : any immutable
            name of child node
        data : tuple or float
            if not is_leaf then tuple (split_col, split_val)
            if is_leaf then float value of y_hat
        is_leaf : bool
            whether or not this node is a terminal leaf

        Returns
        -------
        None
        """
        self.children[key] = Node(data, is_leaf)

    def get_leaf(self, row):
        """ If node is leaf, returns self.data (y_hat), else query get_leaf for
        the appropriate child node and recursively find y_hat

        Parameters
        ----------
        row : numpy array
            features for row of data

        Returns
        -------
        float
            y_hat for row
        """
        if self.is_leaf:
            return self.data
        else:
            if row[self.data[0]] <= self.data[1]:
                return self.children['b'].get_leaf(row)
            else:
                return self.children['a'].get_leaf(row)
