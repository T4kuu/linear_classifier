import numpy as np


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def initialize(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def linear(self, X):
        return X @ self.weights + self.bias

    def predict_proba(self, X):
        z = self.linear(X)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)