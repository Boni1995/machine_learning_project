import pandas as pd
import os

class Mushrooms_dataset:
    """
    This is a data structure consisting in the DataFrame containing data about mushrooms.
    
    Parameters:
        path (str): A path that points where the dataset is stored.
    """
    def __init__(self, path):

        self.path = os.path.join(path, "")


    def import_data(self, filepath):

        data = pd.read_csv(filepath, sep = ";")

        return data


    @property
    def dataset(self):

        data = self.import_data(self.path + "secondary_data.csv")    
            
        return data