# Imports

import numpy as np

# Neural Network Class Definition


class MLP:
    def __init__(
        self,
        neurons_per_layer: list,
        activation_function="relu",
        learning_rate=0.01,
    ):
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.f = None  # Activation Funcion
        self.df = None  # Derivative Activation Function
        self.weights = None
        self.biases = None

        self._activation_function_validation()
        self._initializes_the_activation_function()
        self._initialize_weights_and_biases()

    def train(self, X, y):
        self._input_validation(X)
        self._output_validation(y)

        num_samples = X.shape[1]

        # Forward Pass
        activation = X
        activations = [X]
        pre_activations = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = self.f(z)
            pre_activations.append(z)
            activations.append(activation)

        # Backward Pass
        delta = (activations[-1] - y) * self.df(pre_activations[-1])
        db = [np.sum(delta, axis=1, keepdims=True) / num_samples]
        dw = [np.dot(delta, activations[-2].T) / num_samples]

        for l in range(2, len(self.neurons_per_layer)):
            z = pre_activations[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * self.df(z)
            db.append(np.sum(delta, axis=1, keepdims=True) / num_samples)
            dw.append(np.dot(delta, activations[-l - 1].T) / num_samples)

        db.reverse()
        dw.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]

    def predict(self, X):
        self._input_validation(X)

        signal = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, signal) + b
            signal = self.f(z)

        return signal

    def _activation_function_validation(self):
        allowed_values = ["relu", "sigmoid", "tanh", "leaky", "elu", "swish"]

        if not isinstance(self.activation_function, str):
            raise TypeError(f"The Activation Function must be of type {str}.")
        elif self.activation_function not in allowed_values:
            raise ValueError(
                f"The Activation Function must be one of {allowed_values}."
            )

    def _initializes_the_activation_function(self):
        if self.activation_function == "relu":
            self.f = lambda X: np.maximum(0, X)
            self.df = lambda X: (X > 0).astype(float)

        elif self.activation_function == "sigmoid":
            self.f = lambda X: 1 / (1 + np.exp(-X))
            self.df = lambda X: self.f(X) * (1 - self.f(X))

        elif self.activation_function == "tanh":
            self.f = lambda X: np.tanh(X)
            self.df = lambda X: 1 - np.tanh(X) ** 2

        elif self.activation_function == "leaky":
            alpha = 0.01
            self.f = lambda X: np.where(X > 0, X, alpha * X)
            self.df = lambda X: np.where(X > 0, 1, alpha)

        elif self.activation_function == "elu":
            alpha = 1.0
            self.f = lambda X: np.where(X > 0, X, alpha * (np.exp(X) - 1))
            self.df = lambda X: np.where(X > 0, 1, self.f(X) + alpha)

        elif self.activation_function == "swish":
            sigmoid = lambda X: 1 / (1 + np.exp(-X))
            self.f = lambda X: X * sigmoid(X)
            self.df = lambda X: sigmoid(X) * (1 + X * (1 - sigmoid(X)))

    def _initialize_weights_and_biases(self):
        self.weights = []
        self.biases = []

        for n_in, n_out in zip(self.neurons_per_layer[:-1], self.neurons_per_layer[1:]):
            self.weights.append(np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in))
            if self.activation_function == "relu":
                self.biases.append(np.zeros((n_out, 1)) + 0.01)
            else:
                self.biases.append(np.zeros((n_out, 1)))

    def _input_validation(self, vector):
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"The Input must be a NumPy array {np.ndarray}.")

        elif vector.shape[0] != self.neurons_per_layer[0]:
            raise ValueError(
                f"Input shape {vector.shape} is incompatible with first layer size {self.neurons_per_layer[0]}."
            )

        elif vector.ndim != 2:
            raise ValueError(f"The Input must be a 2D array.")

        elif np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            raise ValueError("Input contains NaN or Inf.")

    def _output_validation(self, vector):
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"The Input must be a NumPy array {np.ndarray}.")

        elif np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            raise ValueError("Input contains NaN or Inf.")
