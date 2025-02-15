import pandas as pd
import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left_leaf=None, right_leaf=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left_leaf = left_leaf
        self.right_leaf = right_leaf
        self.prediction = prediction

    def predict(self, x):
        if self.is_leaf():
            return self.prediction
        if x[self.feature] <= self.threshold:
            return self.left_leaf.predict(x)
        else:
            return self.right_leaf.predict(x)
        
    def is_leaf(self):
        return self.prediction is not None