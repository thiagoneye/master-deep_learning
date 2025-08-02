# Imports

import numpy as np

# Neural Network Class Definition


class MLP:
    def __init__(
        self,
        neurons_per_layer: np.array,
        activation_function="relu",
        learning_rate=0.01,
    ):
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.f = None  # Activation Funcion
        self.df = None  # Derivative Activation Function
        self.weights = None
        # TODO Criar o bias

        self._activation_function_validation(activation_function)
        self._initializes_the_activation_function(activation_function)
        self._initialize_weight_matrix()

    def query(self, X):
        self._signal_validation(X)

        input_signal = X.copy()
        pre_activations_signals = []
        output_signals = []

        for weight in self.weights:
            z_signal = np.dot(weight, input_signal)
            input_signal = self.f(z_signal)
            pre_activations_signals.append(z_signal)
            output_signals.append(input_signal)

        return pre_activations_signals, output_signals

    def train(self, X, y):
        self._signal_validation(y)

        num_layers = len(self.weights)
        pre_activations_signals, output_signals = self.query(X)
        error = y - output_signals[-1]

        # Backpropagation
        for layer_idx in reversed(range(num_layers)):
            z_signal = pre_activations_signals[layer_idx]
            input_signal = X if layer_idx == 0 else output_signals[layer_idx - 1]

            df = self.df(z_signal)
            dL = np.dot(
                (error * df), input_signal
            )  # Partial derivative (gradient) of the loss function with respect to the weights

            self.weights[layer_idx] -= (
                self.learning_rate * dL
            )  # TODO Verificar durante os testes se o sinal irá afetar os resultados

            error = np.dot(self.weights[layer_idx].T, error)

    def predict(self):
        pass

    def _activation_function_validation(self, activation_function):
        allowed_values = ["relu", "sigmoid", "tanh", "leaky", "elu", "swish"]

        if not isinstance(activation_function, str):
            raise TypeError(f"The Activation Function must be of type {str}.")
        elif activation_function not in allowed_values:
            raise ValueError(
                f"The Activation Function must be one of {allowed_values}."
            )

    def _initializes_the_activation_function(self, activation_function):
        if activation_function == "relu":
            self.f = lambda X: np.maximum(0, X)
            self.df = lambda X: np.where(X > 0, 1, 0)

        elif activation_function == "sigmoid":
            self.f = lambda X: 1 / (1 + np.exp(-X))
            self.df = lambda X: self.f(X) * (1 - self.f(X))

        elif activation_function == "tanh":
            self.f = lambda X: np.tanh(X)
            self.df = lambda X: 1 - np.tanh(X) ** 2

        elif activation_function == "leaky":
            alpha = 0.01
            self.f = lambda X: np.where(X > 0, X, alpha * X)
            self.df = lambda X: np.where(X > 0, 1, alpha)

        elif activation_function == "elu":
            alpha = 1.0
            self.f = lambda X: np.where(X > 0, X, alpha * (np.exp(X) - 1))
            self.df = lambda X: np.where(X > 0, 1, self.f(X) + alpha)

        else:
            # swish
            sigmoid = lambda X: 1 / (1 + np.exp(-X))
            self.f = lambda X: X * sigmoid(X)
            self.df = lambda X: sigmoid(X) * (1 + X * (1 - sigmoid(X)))

    def _initialize_weight_matrix(self):
        structure_of_the_weight_matrix = np.stack(
            (self.neurons_per_layer[:-1], self.neurons_per_layer[1:]), axis=1
        )

        self.weights = [
            np.random.rand(*size) * 2 - 1 for size in structure_of_the_weight_matrix
        ]

    def _signal_validation(self, vector):
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"The Input must be a NumPy array {np.ndarray}.")

        elif vector.ndim != 2:
            raise ValueError(f"The Input must be a 2D array.")

        elif np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            raise ValueError("Input contains NaN or Inf.")

    def _loss_function_validation(self, loss_function):
        allowed_values = ["mse", "mae"]

        if not isinstance(loss_function, str):
            raise TypeError(f"The Loss Function must be of type {str}.")
        elif loss_function not in allowed_values:
            raise ValueError(f"The Loss Function must be one of {allowed_values}.")

    def _initializes_the_loss_function(self, loss_function):
        if loss_function == "mse":
            self.J = lambda target, output: np.sum((target - output) ** 2) / len(target)

        elif loss_function == "mae":
            self.J = lambda target, output: np.sum(np.abs(target - output)) / len(
                target
            )
