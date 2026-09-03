import numpy as np


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


def gradients(X, y, probabilities):
    n_samples = X.shape[0]

    error = probabilities - y

    dw = (X.T @ error) / n_samples
    db = np.sum(error) / n_samples

    return dw, db