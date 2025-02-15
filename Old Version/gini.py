import pandas as pd
import numpy as np

class Gini:

    def __init__(self, left_leaf, right_leaf):

        self.left_leaf = left_leaf
        self.right_leaf = right_leaf
    
    def gini_function(self, data):
        
        labels = data[:, 0]
        _, counts = np.unique(labels, return_counts=True)
        total_count = len(labels)
        p = counts / total_count

        gini = 1 - np.sum(p ** 2)

        return gini

    def overall_gini(self):

        n = len(self.left_leaf) + len(self.right_leaf)
        p_left = len(self.left_leaf) / n
        p_right = len(self.right_leaf) / n
        
        overall_gini =  (p_left * self.gini_function(self.left_leaf) + p_right * self.gini_function(self.right_leaf))
        
        return overall_gini