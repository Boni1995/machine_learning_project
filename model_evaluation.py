import numpy as np

class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp, self.tn, self.fp, self.fn = self.confusion_matrix()

    def confusion_matrix(self):
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        return tp, tn, fp, fn

    def precision_score(self):
        if (self.tp + self.fp) == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def recall_score(self):
        if (self.tp + self.fn) == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        precision = self.precision_score()
        recall = self.recall_score()
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)

    def evaluate(self):
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision_score(),
            'recall': self.recall_score(),
            'f1_score': self.f1_score(),
            'confusion_matrix': {
                'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn
            }
        }