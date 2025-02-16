import pandas as pd
import numpy as np
from criterions import information_gain
from tree_node import TreeNode
from joblib import Parallel, delayed

class DecisionTree:
    def __init__(self, max_depth=None, min_samples=2, criterion='gini'):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples = min_samples

    def get_params(self):
        return {
            'max_depth': self.max_depth,
            'min_samples': self.min_samples,
            'criterion': self.criterion
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _best_split(self, x, y):
        best_feature, best_threshold = None, None
        best_score = -float('inf')
        m, n = x.shape

        def evaluate_split(feature):
            local_best_score = -float('inf')
            local_best_threshold = None

            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = x[:, feature] > threshold

                if sum(left_mask) < self.min_samples or sum(right_mask) < self.min_samples:
                    continue

                score = information_gain(y, y[left_mask], y[right_mask], self.criterion)

                if score > local_best_score:
                    local_best_score = score
                    local_best_threshold = threshold

            return feature, local_best_threshold, local_best_score
        
        results = Parallel(n_jobs=-1)(delayed(evaluate_split)(feature) for feature in range(n))

        for feature, threshold, score in results:
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def _build_tree(self, x, y, depth=0):
        num_samples, _ = x.shape
        num_labels = len(np.unique(y))

        if num_labels == 1 or (self.max_depth is not None and depth == self.max_depth) or num_samples < self.min_samples:
            return TreeNode(prediction=self._most_common_label(y))

        feature, threshold = self._best_split(x, y)
        if feature is None:
            return TreeNode(prediction=self._most_common_label(y))

        left_mask = x[:, feature] <= threshold
        right_mask = x[:, feature] > threshold

        left_child = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(x[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature, threshold, left_child, right_child)

    def fit(self, x, y):
        self.root = self._build_tree(x, y)

    def predict(self, x):
        return np.array([self.root.predict(sample) for sample in x])

    def evaluate(self, x, y):
        predictions = self.predict(x)
        return np.sum(predictions != y) / len(y)