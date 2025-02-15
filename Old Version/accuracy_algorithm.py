import pandas as pd
import numpy as np

class TreeAccuracy:

    def __init__(self, data, tree):
        self.data = data
        self.tree = tree


    def test_classification(self, sample):
        question = list(self.tree.keys())[0]
        attribute, comparison, value, _, _, _, _ = question.split(" ")

        # ask question
        if comparison == "<=":
            if sample[attribute] <= float(value):
                answer = self.tree[question][0]

            else:
                answer = self.tree[question][1]
        
        # feature is categorical
        else:

            if str(sample[attribute]) == value:
                answer = self.tree[question][0]

            else:
                answer = self.tree[question][1]

        # base case
        if not isinstance(answer, dict):
            return answer
        
        # recursive part
        else:
            residual_tree = answer

            return TreeAccuracy(data = self.data, tree = residual_tree).test_classification(sample)


    def tree_accuracy(self):

        self.data["classification"] = self.data.apply(self.test_classification, axis=1)

        self.data["classification_correct"] = self.data["classification"] == self.data["class"]
        
        accuracy = self.data["classification_correct"].mean()
        
        return accuracy