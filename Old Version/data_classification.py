import pandas as pd
import numpy as np


class Classification:

    def __init__(self, data):
        self.data = data


    def classify_data(self):
        
        labels = self.data[:, 0]
        unique_label, count_unique_label = np.unique(labels, return_counts=True)
        index = count_unique_label.argmax()
        
        classification = unique_label[index]
        
        return classification


    def attribute_type(self):
        
        attribute_types = []
        treshold = 20

        _, n_columns = self.data.shape

        for attribute in range(0, n_columns):
            values = self.data[:, attribute]
            unique_values = len(set(values))
            first_value = values[0]

            if (isinstance(first_value, str)) or (unique_values <= treshold):
                attribute_types.append("categorical")
                
            else:
                attribute_types.append("continuous")
        
        return attribute_types