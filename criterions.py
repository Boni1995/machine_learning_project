import pandas as pd
import numpy as np



def gini(y):
    _, counts = np.unique(y, return_counts=True)
    total_count = len(y)
    p = counts / total_count
    return 1 - np.sum(p ** 2)


def entropy(y): 
    _, counts = np.unique(y, return_counts=True)
    total_count = len(y)
    p = counts / total_count
    return -np.sum(p * np.log2(p))


def misclassification_error(y):
    _, counts = np.unique(y, return_counts=True)
    total_count = len(y)
    most_common_label_count = np.max(counts)
    return 1 - most_common_label_count / total_count


def impurity(y, metric="gini"):
    if metric == "gini":
        impurity = gini(y)
    elif metric == "entropy":
        impurity = entropy(y)
    elif metric == "misclassification_error":
        impurity = misclassification_error(y)
    else:
        raise ValueError("Use 'gini', 'entropy' or 'misclassification_error'.")
    return impurity


def information_gain(y, y_left, y_right, metric="gini"):
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    impurity_y = impurity(y, metric)
    impurity_left = impurity(y_left, metric)
    impurity_right = impurity(y_right, metric)

    return impurity_y - (n_left / n) * impurity_left - (n_right / n) * impurity_right