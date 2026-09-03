class GradientDescent:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model, dw, db):
        model.weights -= self.learning_rate * dw
        model.bias -= self.learning_rate * db