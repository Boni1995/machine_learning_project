import pandas as pd
import numpy as np

class Criterias:

    def __init__(self, left_leaf, right_leaf):

        self.left_leaf = left_leaf
        self.right_leaf = right_leaf
    
    def probability(self, data):
        labels = data[:, 0]
        _, counts = np.unique(labels, return_counts=True)
        total_count = len(labels)
        p = counts / total_count

        return p

    def criteria (self, data, criterion):

        p = self.probability(data)

        if criterion == 'gini':

            gini = 2*p*(1-p)

            return gini

        elif criterion == 'entropy':

            if np.any(p == 0):

                return 0
            
            else:
                entropy = -p/2 * np.log2(p) - (1-p)/2 * np.log2(1-p)

            return entropy
        
        elif criterion == 'quadratic_dev':

            quadratic_dev = np.sqrt(p * (1-p))

            return quadratic_dev
        
        else:

            raise ValueError(f"Unknown criterion: {criterion}")

    def overall_criteria(self, criterion):

        n = len(self.left_leaf) + len(self.right_leaf)
        p_left = len(self.left_leaf) / n
        p_right = len(self.right_leaf) / n
        
        overall_criteria =  (p_left * self.criteria(self.left_leaf, criterion) + p_right * self.criteria(self.right_leaf, criterion))
        
        return overall_criteria