# master-deep_learning

This repository contains implementations of various algorithms and concepts covered in the context of a Deep Learning discipline. The goal is to provide practical examples and a deeper understanding of the theoretical foundations taught in the course.

## Table of Contents

- Overview
- Implemented Algorithms and Concepts
- Installation
- Usage

## Overview

This repository serves as a practical companion to the theoretical knowledge acquired in a Deep Learning discipline. It includes hands-on implementations of fundamental algorithms, neural network architectures, optimization techniques, and other key concepts. Each implementation aims to be clear, well-documented, and easy to understand, facilitating both learning and experimentation.

## Implemented Algorithms and Concepts

Here's a list of some of the algorithms and concepts you'll find implemented in this repository (this list will be updated as more implementations are added):

- **Perceptron**: A foundational neural network model.
- **Multi-Layer Perceptron (MLP)**: Basic feedforward neural networks.
- **Backpropagation Algorithm**: The core algorithm for training neural networks.
- **Activation Functions**: Implementations of common activation functions like ReLU, Sigmoid, and Tanh.
- **Loss Functions**: Examples of loss functions such as Mean Squared Error (MSE) and Cross-Entropy.
- **Optimization Algorithms**: Stochastic Gradient Descent (SGD), Adam, RMSprop.
- **Convolutional Neural Networks (CNNs)**: Basic architectures for image processing.
- **Recurrent Neural Networks (RNNs)**: Simple RNNs for sequential data.
- **Batch Normalization**: Implementation of batch normalization layers.
- **Dropout**: Regularization technique for neural networks.

## Installation

To get started with these implementations, you'll need to clone the repository and install the necessary dependencies.

1. Clone the repository:

```
git clone git@github.com:thiagoneye/master-deep_learning.git
cd master-deep_learning
```

2. Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Each algorithm or concept typically resides in its own directory or dedicated Python file. You can navigate to the specific implementation you're interested in and run the associated script.

For example, to run the Perceptron example:

```
cd perceptron
python perceptron_example.py
```
