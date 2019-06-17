class Node:
    def __init__(self,name,index,parent_index=None):
        self.name = name
        self.index = index
        self.parent_index = parent_index
        self.pos = self.parent = None
