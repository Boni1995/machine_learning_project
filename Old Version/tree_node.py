import pandas as pd
import numpy as np

from gini import Gini
from data_classification import Classification


class TreeNode:

    def __init__(self, data, criteria = None, threshold = None):
        self.data = data
        self.criteria = criteria
        self.threshold = threshold
        self.left_leaf = None
        self.right_leaf = None
    

    def is_leaf (self):

        labels = self.data[:, 0]
        unique_variable = np.unique(labels)

        if len(unique_variable) == 1:
            return True
        
        else:
            return False
    

    def all_potential_splits(self):
        
        potential_splits = {}
        _, n_columns = self.data.shape

        for column_index in range(1, n_columns):
            values = self.data[:, column_index]
            unique_values = np.unique((values[values != "na"]))

            potential_splits[column_index] = unique_values
        
        return potential_splits


    def split_data(self):

        split_column_values = self.data[:, self.criteria]
        type_attribute = Classification(self.data).attribute_type()
        type_of_attribute = type_attribute[self.criteria]

        # Continuous
        if type_of_attribute == "continuous":
            left_leaf = self.data[split_column_values <= self.threshold]
            right_leaf = self.data[split_column_values >  self.threshold]
        
        # Categorical   
        else:
            left_leaf = self.data[split_column_values == self.threshold]
            right_leaf = self.data[split_column_values != self.threshold]
        
        return left_leaf, right_leaf


    def best_split(self, potential_splits):
        
        base_gini = float('inf')

        for column_index in potential_splits:
            for value in potential_splits[column_index]:

                self.left_leaf, self.right_leaf = TreeNode(self.data, criteria = column_index, threshold = value).split_data()
                current_overall_gini = Gini(self.left_leaf, self.right_leaf).overall_gini()

                if current_overall_gini <= base_gini:
                    base_gini = current_overall_gini
                   
                    best_attribute = column_index
                    best_value = value
        
        return best_attribute, best_value, current_overall_gini