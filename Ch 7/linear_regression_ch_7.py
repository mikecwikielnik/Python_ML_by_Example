"""
Implementing linear regression from scratch

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 228). Packt Publishing. Kindle Edition. 
"""

import numpy as np

def compute_prediction(X, weights):
    """
    Compute the prediction y_hat based on current weights
    """

    predictions = np.dot(X, weights)
    return predictions

