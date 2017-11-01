class Tree:

    def __init__(self):
        self.tree = {'root':[]}

    def add_node(self, keys, value, tree, i=1):
        """

        Parameters
        ----------
        keys :  list
            keys for traversing tree
        value : list, float, or int
            value to be inserted into tree
        tree : nested dict
            tree of variable depth and general structure:
            {key: [n1, n2, ..., {key: [n1, n2, ..., {key: y_hat}]}
        i : int
            incrementer for tracking depth

        Returns
        -------
        None
        """
        if i == len(keys):
            tree[keys[i]] = value
        else:
            self.add_node(keys=keys,
                          tree=tree[keys[i]][-1],
                          i=i + 1)

    def lookup_value(self, keys, tree, i=1):
        """

        Parameters
        ----------
        keys : list
            keys for traversing tree
        tree : nested dict
            tree of variable depth and general structure:
            {key: [n1, n2, ..., {key: [n1, n2, ..., {key: y_hat}]}
        i : int
            incrementer for tracking depth

        Returns
        -------
        value for deepest branch of tree provided by keys
        """
        if i == len(keys):
            return tree[keys[i]]
        else:
            return self.lookup_value(keys=keys,
                                     tree=tree[keys[i]][-1],
                                     i=i + 1)
