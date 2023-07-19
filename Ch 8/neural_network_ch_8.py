"""
Predicting Stock Prices with Artificial Neural Networks

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 253). Packt Publishing. Kindle Edition. 
"""

# implementing neural networks from scratch

# sigmoid as the activation function in this example

import numpy as np

# define the sigmoid fn and the derivative fn

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

