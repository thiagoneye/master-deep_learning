# Make Your Own Neural Network

This directory contains my implementation of a neural network built **entirely from scratch**, without relying on popular deep learning frameworks like TensorFlow, Keras, or PyTorch. The goal here is to deeply understand the fundamental mathematics and algorithms that power neural networks, by implementing each component step-by-step.

-----

## Inspiration and References

This implementation is heavily inspired by and based on the principles described in two excellent resources:

  * **"Make Your Own Neural Network" by Tariq Rashid (2016)**: This book provides a clear, step-by-step guide to building a simple neural network in Python, making complex concepts accessible.
  * **"Neural Networks from Scratch in Python" by Harrison Kinsley (2020)**: This resource offers a more comprehensive dive, covering various activation functions, optimizers, and layers, all implemented without external libraries beyond basic numerical computation.

-----

## Contents

Within this folder, you'll find Python scripts (`.py`) and Jupyter Notebooks (`.ipynb`) that progressively build up the neural network. Key components you can expect to find include:

  * **`neural_network.py`**: The core implementation of the neural network class, containing methods for:
      * **Initialization**: Setting up layers, weights, and biases.
      * **Forward Propagation**: Calculating outputs from inputs.
      * **Backward Propagation (Backpropagation)**: Computing gradients for weight and bias updates.
      * **Training**: Iterating through data, performing forward and backward passes, and updating parameters.
      * **Prediction**: Using the trained model to make predictions on new data.
  * **`layers.py`**: Definitions for different types of layers (e.g., dense/fully connected layers).
  * **`activations.py`**: Implementations of various activation functions (e.g., sigmoid, ReLU, softmax).
  * **`optimizers.py`**: Simple gradient descent optimizer.
  * **`loss_functions.py`**: Implementation of loss functions (e.g., mean squared error, cross-entropy).
  * **Example usage scripts**: Demonstrations of how to instantiate, train, and test the neural network on simple datasets.

-----

## Getting Started

To explore this implementation, you'll primarily need Python and NumPy.

1.  **Clone the repository**:
    ```bash
    git clone git@github.com:thiagoneye/master-deep_learning.git
    ```
2.  **Navigate to this directory**:
    ```bash
    cd master-deep_learning/MakeYourOwnNeuralNetwork
    ```
3.  **Install dependencies**:
    ```bash
    pip install numpy pandas matplotlib
    ```

-----

## Why Implement from Scratch?

Building a neural network from the ground up offers several benefits:

  * **Deeper Understanding**: It demystifies the "black box" nature of neural networks, revealing how weights, biases, activations, and gradients interact.
  * **Problem-Solving Skills**: It hones your ability to translate mathematical concepts into executable code.
  * **Debugging Prowess**: Understanding the internal workings makes it easier to debug issues when working with frameworks.
  * **Foundation for Custom Architectures**: This knowledge is invaluable for designing and implementing novel neural network architectures in the future.
