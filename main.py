import csv

import numpy as np

from src.model import LogisticRegression
from src.loss import binary_cross_entropy, gradients
from src.optimizer import GradientDescent
from src.preprocessing import train_test_split, standardize
from src.metrics import accuracy, confusion_matrix


DATA_PATH = "data/data.csv"

LEARNING_RATE = 0.01
EPOCHS = 1000


def load_data(path):
    data = []

    with open(path, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            try:
                values = [float(value) for value in row]
                data.append(values)
            except ValueError:
                continue

    data = np.array(data)

    X = data[:, :-1]
    y = data[:, -1].astype(int)

    return X, y


def train(model, optimizer, X, y, epochs):
    history = []

    for epoch in range(epochs):
        probabilities = model.predict_proba(X)

        loss = binary_cross_entropy(y, probabilities)

        dw, db = gradients(X, y, probabilities)

        optimizer.step(model, dw, db)

        history.append(loss)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:4d}/{epochs} | "
                f"Loss: {loss:.6f}"
            )

    return history


def main():
    print("Loading dataset...")

    X, y = load_data(DATA_PATH)

    print(f"Samples:  {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        seed=42
    )

    X_train, X_test = standardize(
        X_train,
        X_test
    )

    model = LogisticRegression()
    model.initialize(X_train.shape[1])

    optimizer = GradientDescent(
        learning_rate=LEARNING_RATE
    )

    print("\nTraining...\n")

    history = train(
        model,
        optimizer,
        X_train,
        y_train,
        EPOCHS
    )

    print("\nEvaluation:")

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy(
        y_test,
        test_predictions
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions
    )

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nLearned weights:")
    print(model.weights)

    print(f"\nLearned bias: {model.bias:.6f}")

    print(f"\nFinal training loss: {history[-1]:.6f}")


if __name__ == "__main__":
    main()