import pandas as pd
import numpy as np
import random


class TrainTestSplit:

    def __init__(self, data, test_size, seed):

        self.data = data
        self.test_size = test_size
        self.seed = seed

        if isinstance(self.test_size, float):
            self.test_size = round(self.test_size * len(self.data))
        
        if self.test_size < 0 or self.test_size > len(self.data):
            raise ValueError("test_size must be between 0 and the length of the data.")

    def split(self):

        random.seed(self.seed)

        indices = self.data.index.tolist()
        test_indices = random.sample(population = indices, k = self.test_size)

        test_set = self.data.loc[test_indices]
        train_set = self.data.drop(test_indices)

        return train_set, test_set