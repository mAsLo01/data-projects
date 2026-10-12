"""A small fully connected neural network implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the sigmoid while avoiding overflow in exp."""
    clipped_values = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


class NeuralNetwork:
    """Binary classifier with one hidden layer and sigmoid activations."""

    def __init__(
        self,
        n_hidden: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        if n_hidden <= 0:
            raise ValueError("n_hidden must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_state)
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    def _initialize_parameters(self, n_features: int) -> None:
        # Xavier initialization keeps sigmoid activations away from saturation.
        hidden_limit = np.sqrt(6.0 / (n_features + self.n_hidden))
        output_limit = np.sqrt(6.0 / (self.n_hidden + 1))

        self.weights_hidden = self.rng.uniform(
            -hidden_limit,
            hidden_limit,
            size=(n_features, self.n_hidden),
        )
        self.bias_hidden = np.zeros((1, self.n_hidden))
        self.weights_output = self.rng.uniform(
            -output_limit,
            output_limit,
            size=(self.n_hidden, 1),
        )
        self.bias_output = np.zeros((1, 1))

    @staticmethod
    def _binary_cross_entropy(
        targets: NDArray[np.float64],
        probabilities: NDArray[np.float64],
    ) -> float:
        epsilon = 1e-12
        probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(
                targets * np.log(probabilities)
                + (1.0 - targets) * np.log(1.0 - probabilities)
            )
        )

    def _forward(
        self,
        features: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        hidden = sigmoid(features @ self.weights_hidden + self.bias_hidden)
        output = sigmoid(hidden @ self.weights_output + self.bias_output)
        return hidden, output

    def _backward(
        self,
        features: NDArray[np.float64],
        targets: NDArray[np.float64],
        hidden: NDArray[np.float64],
        output: NDArray[np.float64],
    ) -> None:
        n_samples = features.shape[0]

        # For sigmoid output and binary cross-entropy, dL/dz = output - target.
        output_delta = (output - targets) / n_samples
        gradient_weights_output = hidden.T @ output_delta
        gradient_bias_output = np.sum(output_delta, axis=0, keepdims=True)

        hidden_delta = (
            output_delta @ self.weights_output.T
        ) * hidden * (1.0 - hidden)
        gradient_weights_hidden = features.T @ hidden_delta
        gradient_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)

        self.weights_output -= self.learning_rate * gradient_weights_output
        self.bias_output -= self.learning_rate * gradient_bias_output
        self.weights_hidden -= self.learning_rate * gradient_weights_hidden
        self.bias_hidden -= self.learning_rate * gradient_bias_hidden

    def fit(
        self,
        features: ArrayLike,
        targets: ArrayLike,
        epochs: int = 1_000,
        verbose_every: int | None = 100,
    ) -> "NeuralNetwork":
        """Train the network using full-batch gradient descent."""
        features_array = np.asarray(features, dtype=np.float64)
        targets_array = np.asarray(targets, dtype=np.float64).reshape(-1, 1)

        if features_array.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if features_array.shape[0] != targets_array.shape[0]:
            raise ValueError("features and targets must contain the same samples")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        self.feature_mean = features_array.mean(axis=0)
        self.feature_std = features_array.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        normalized_features = (
            features_array - self.feature_mean
        ) / self.feature_std

        self._initialize_parameters(normalized_features.shape[1])
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(1, epochs + 1):
            hidden, probabilities = self._forward(normalized_features)
            loss = self._binary_cross_entropy(targets_array, probabilities)
            predictions = (probabilities >= 0.5).astype(np.int64)
            accuracy = float(np.mean(predictions == targets_array))

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            self._backward(
                normalized_features,
                targets_array,
                hidden,
                probabilities,
            )

            if verbose_every and (
                epoch == 1 or epoch % verbose_every == 0 or epoch == epochs
            ):
                print(
                    f"Epoch {epoch:4d}/{epochs}: "
                    f"loss={loss:.4f}, accuracy={accuracy:.4f}"
                )

        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return probabilities for the positive class."""
        if not hasattr(self, "feature_mean"):
            raise RuntimeError("The model must be fitted before prediction")

        features_array = np.asarray(features, dtype=np.float64)
        normalized_features = (
            features_array - self.feature_mean
        ) / self.feature_std
        _, probabilities = self._forward(normalized_features)
        return probabilities.ravel()

    def predict(self, features: ArrayLike) -> NDArray[np.int64]:
        """Return binary class labels."""
        return (self.predict_proba(features) >= 0.5).astype(np.int64)
