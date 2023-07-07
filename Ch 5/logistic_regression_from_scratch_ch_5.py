"""
Training a logistic regression model

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 158). Packt Publishing. Kindle Edition. 
"""

import numpy as np

def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

# Gradient descent based logistic regression from scratch

def compute_prediction(X, weights):
    """
    Compute the prediction y_hat based on current weights
    """

    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions
