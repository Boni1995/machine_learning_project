import pandas as pd
import numpy as np

class Tuning:

    def param_grid():
        return {
        'max_depth': [3, 5, 10],
        'min_samples': [300, 500, 1000],
        'criterion': ['gini', 'entropy', 'quadratic_dev']
        }