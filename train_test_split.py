import pandas as pd
import numpy as np
import random

def split(X, y, test_size=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test