import numpy as np


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std = np.where(std == 0, 1, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled