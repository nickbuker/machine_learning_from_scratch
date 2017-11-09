class Node:
    def __init__(self, splits, data):
        self.splits = splits
        self.data = data
        self.children = {}

    def add_child(self, key, splits, data):
        self.children[key] = Node(splits, data)
        