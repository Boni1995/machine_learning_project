import pandas as pd
import numpy as np

from tree_node import TreeNode
from data_classification import Classification


class TreePredictor:

    def __init__(self, data, counter, min_samples, max_depth):
        self.data = data
        self.counter = counter
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.type_attribute = None
        self.column_names = None


    def tree_predictor(self):

        # data preparations
        if self.counter == 0:
            global column_names, type_attribute
            column_names = self.data.columns
            type_attribute = Classification(self.data.values).attribute_type()
            data = self.data.values
        else:
            data = self.data           
        
        n_sample = len(data)

        # Stop criterias
        if (TreeNode(data).is_leaf()) or (len(data) < self.min_samples) or (self.counter == self.max_depth):
            classification = Classification(data).classify_data()
            
            return classification

        
        # Recursive part
        else:    
            self.counter += 1

            potential_splits = TreeNode(data).all_potential_splits()
            best_criteria, threshold, gini = TreeNode(data).best_split(potential_splits)
            left_split, right_split = TreeNode(data, criteria = best_criteria, threshold = threshold).split_data()
            
            # Control of empty data after split
            if len(left_split) == 0 or len(right_split) == 0:
                classification = Classification(data).classify_data()

                return classification
            
            # Question
            attribute_name = column_names[best_criteria]
            type_of_attribute = type_attribute[best_criteria]
            
            if type_of_attribute == "continuous":
                question = "{} <= {} (gini: {}, n: {})".format(attribute_name, threshold, round(gini, 2), n_sample)
                
            # If feature is categorical
            else:
                question = "{} = {} (gini: {}, n: {})".format(attribute_name, threshold, round(gini, 2), n_sample)
            
            # Instantiate sub-tree
            sub_tree = {question: []}

            # Find answers (recursion)
            yes = TreePredictor(left_split, self.counter, self.min_samples, self.max_depth).tree_predictor()
            no = TreePredictor(right_split, self.counter, self.min_samples, self.max_depth).tree_predictor()
            
            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (min_samples or max_depth base case).
            if yes == no:
                sub_tree = yes
            else:
                sub_tree[question].append(yes)
                sub_tree[question].append(no)
            
            return sub_tree