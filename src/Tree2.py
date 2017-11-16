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
