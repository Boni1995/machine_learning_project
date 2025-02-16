import pandas as pd
import os

class Mushrooms_dataset:

    def __init__(self, path):

        self.path = os.path.join(path, "")


    def import_data(self, filepath):

        data = pd.read_csv(filepath, sep = ";")

        return data


    @property
    def dataset(self):

        data = self.import_data(self.path + "secondary_data.csv")    
            
        return data