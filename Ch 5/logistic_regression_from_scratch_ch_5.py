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

# we connect all these functions to the model training function by doing the following:
# 1) updating the weights vector in each iteration
# 2) printing the current cost for every k iterations to ensure cost is decreasing

def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept = False):
    """Train a logistic regression model
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        max_iter (int, number of iterations)
        learning_rate(float)
        fit_intercept (bool, with an intercept w0 or not)
    Returns:
        numpy.ndarray, learned weights
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        # check the cost for every 100 (for ex) iterations
        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

# finally, we predict the results of new inputs using the trained model as follows:

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

# an example

X_train = np.array([[6, 7],
                    [2, 4],
                    [3, 6],
                    [4, 7],
                    [1, 6],
                    [5, 2],
                    [2, 0],
                    [6, 3],
                    [4, 1],
                    [7, 2]])

y_train = np.array([0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1])

weights = train_logistic_regression(X_train, y_train, max_iter=1000, learning_rate=0.1, fit_intercept=True)

# decreasing cost means that the model is being optimized over time

X_test = np.array([[6, 1],
                   [1, 3],
                   [3, 1],
                   [4, 5]])

predictions = predict(X_test, weights)
predictions