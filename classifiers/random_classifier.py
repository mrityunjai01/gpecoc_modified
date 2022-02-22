import numpy as np
from sklearn.base import BaseEstimator

class RandomClassifier(BaseEstimator):
    """
        classifies to a random class.
    """
    def __init__(self):
        pass
        # self._estimator_type="regressor"

    def fit(self, X, y):
        self.labels = np.unique(y)
        return self

    def predict(self, X):
        return np.random.choice(self.labels, size=X.shape[0])

    def decision_function(self, X):
        n_classes = 2
        return np.zeros((X.shape[0], ))
