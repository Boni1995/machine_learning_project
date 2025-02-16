import numpy as np
import random
from train_test_split import split
from tree_algorithm import DecisionTree 
from joblib import Parallel, delayed

class RandomizedSearchCV:
    
    def __init__(self, param_grid, x_train, y_train, n_iter=10, cv=5, scoring='accuracy', seed=None, n_jobs=-1):
        self.param_grid = param_grid
        self.x_train = x_train
        self.y_train = y_train
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.seed = seed
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_score = -1

        if self.seed:
            np.random.seed(self.seed)

    def evaluate_fold(self, params, i):
        """Evaluates a single fold of cross-validation."""
        x_train_fold, x_val_fold, y_train_fold, y_val_fold = split(self.x_train, self.y_train, test_size=1/self.cv,
                                                                    seed = i)

        tree = DecisionTree(**params)
        tree.fit(x_train_fold, y_train_fold)
        y_pred = tree.predict(x_val_fold)

        return np.mean(y_val_fold == y_pred)


    def fit(self):
        for _ in range(self.n_iter):

            params = {key: random.choice(values) for key, values in self.param_grid.items()}

            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.evaluate_fold)(params, i) for i in range(self.cv)
            )
            mean_score = np.mean(scores)

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params

        return self.best_params, self.best_score