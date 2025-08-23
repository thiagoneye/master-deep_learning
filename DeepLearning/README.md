# Deep Learning Coursework

This directory contains all the files, assignments, and projects completed as part of my Deep Learning course during my Master's program. It serves as a comprehensive record of the concepts learned and practical applications developed throughout the course.

## Contents

Within this folder, you'll find a variety of materials, likely organized into subdirectories for each assignment or project. Expect to see:

* **Jupyter Notebooks** (.ipynb): These will contain the code, explanations, and visualizations for various assignments and practical exercises.
* **Python Scripts** (.py): Standalone scripts for models, data processing, or utility functions.
* **Datasets**: Smaller datasets used for specific exercises.
* **Reports/Documentation**: Any written reports, analyses, or detailed explanations accompanying the code.

## Key Topics Covered

The coursework in this directory generally covers fundamental and advanced topics in Deep Learning, which may include:

#### Foundational Concepts

* **Fundamentals of Neural Networks**: Learn the building blocks, from simple perceptrons to more complex networks. You'll explore different activation functions that introduce non-linearity and master the core learning algorithm, backpropagation, which enables the network to adjust its weights and biases.

* **Optimization Algorithms**: Dive into the methods that make neural networks learn efficiently, such as Stochastic Gradient Descent (SGD), Adam, and RMSprop, which help find the optimal set of weights to minimize loss.

* **Regularization Techniques**: Understand how to prevent your models from overfitting, or memorizing the training data, using strategies like Dropout and L1/L2 regularization to improve generalization to new data.

* **Frameworks**: Gain hands-on experience by applying these concepts using popular deep learning frameworks like TensorFlow and Keras, which simplify the process of building and training models.

#### Architectures for Different Data Types

* **Convolutional Neural Networks (CNNs)**: Designed for grid-like data such as images, you'll study foundational architectures like LeNet, AlexNet, VGG, and ResNet. These models are crucial for a wide range of computer vision tasks, including image classification.

* **Recurrent Neural Networks (RNNs)**: For tasks involving sequential data like text, time series, or audio, you'll work with RNNs. This includes advanced architectures like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, which were developed to overcome the vanishing gradient problem and better capture long-range dependencies in data. GRUs are a simpler, faster alternative to LSTMs with comparable performance on many tasks.

* **Autoencoders**: Explore a type of neural network designed for unsupervised learning. An autoencoder learns to compress data into a low-dimensional representation (encoding) and then reconstruct the original data from that representation (decoding). They are widely used for tasks like dimensionality reduction, feature extraction, and anomaly detection.

* **Transformers**: This revolutionary architecture has become the new standard for sequential data, particularly in Natural Language Processing (NLP). Unlike RNNs, Transformers process entire sequences at once using a powerful self-attention mechanism, which allows them to capture the relationships between all elements in a sequence, regardless of their position. This parallel processing makes them far more scalable and efficient than LSTMs and GRUs for very large datasets.

* **Generative Adversarial Networks (GANs)**: Learn the principles of these powerful generative models, which consist of two competing networks, a generator and a discriminator, that work together to create realistic new data.

#### Advanced Topics & Applications

* **Large Language Models (LLMs)**: Understand the architecture and training principles behind the current generation of state-of-the-art language models like GPT and BERT. These massive, pre-trained Transformer-based models are capable of a wide range of tasks, from text generation and translation to complex reasoning and summarization.

* **Reinforcement Learning (RL)**: Shift from predictive modeling to decision-making. In RL, an "agent" learns to make a sequence of decisions in an environment to maximize a cumulative reward. You'll explore the core concepts of agents, environments, states, actions, and rewards, and see how RL is used to train systems for tasks like game-playing and robotics.

* **Transfer Learning**: Discover how to leverage knowledge gained from a pre-trained model on one task to accelerate learning on a new, related task. This is a crucial technique for training effective models with limited data.

## How to Navigate

To explore the content, simply browse the subdirectories. Each major assignment or project should have its own dedicated folder. Look for .ipynb files for executable code and explanations, and any accompanying README.md files within subfolders for specific instructions or context.

## Dependencies

The specific dependencies will vary depending on the project. However, common libraries you'll likely need include:

* `tensorflow`
* `keras`
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `jupyter` (to run the notebooks)

You can usually install these using pip:

```bash
pip install tensorflow torch keras numpy pandas matplotlib scikit-learn
```
