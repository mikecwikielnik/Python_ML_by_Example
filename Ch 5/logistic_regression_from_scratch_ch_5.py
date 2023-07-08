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

def update_weights_gd(X_train, y_train, weights, learning_rate):
    """
    update weights by one step
    """
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    """
    compute the cost J(w)
    """
    predictions = compute_prediction(X, weights)
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    return cost

