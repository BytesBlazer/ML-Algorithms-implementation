import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predictions(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x1):
        distances = [euclidean_distance(x1, x2) for x2 in self.x_train]

        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_labels[i] for i  in k_indices]

        majority = Counter(k_labels).most_common()

        return majority




