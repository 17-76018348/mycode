class Node:
    def __init__(self, data, parent):
        self.data = data
        self.left = self.right = None
        self.parent = parent
    def get_left(self):
        return self.left
    def get_right(self):
        return self.right
    def get_data(self):
        return self.data
    def get_parent(self):
        return self.parent
    def set_left(self, left):
        self.left = left
    def set_right(self, right):
        self.right = right
    def set_data(self, data):
        self.data = data
    def set_parent(self,parent):
        self.parent = parent
class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, data, node):
        if self.root is None:
            self.root = Node(data, None)
            return 
        if data == node.get_data
            return
        if data == node.get_data():
            return
        if data > node.get_data():
            if node.get_right() is None:
                node.set_right(Node(data, node))
            else:
                self.insert(data, node.get_RHS())
        if data < node.get_data():
            if node.get_left() is None:
                node.set_left(Node(data, node))
            else:
                self.insert(data, node.get_left)
        return

    def search(self, data, None):
        if data == node.get_data():
            return True
        if data > node.get_data():
            if node.get_right is None:
                return False
            else:
                return self.search(data,node.get_right())
        if data < node.get_data():
            if node.get_left() is None:
                return False
            else:
                return self.search(data, node.get_left())

#%%

