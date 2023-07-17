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

def update_weights_gd(X_train, y_train, weights, learning_rate):
    """
    Update weights by one step and return updated weights
    """

    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    """
    Compute the cost J(w)
    """

    predictions = compute_prediction(X, weights)
    cost = np.mean((predictions - y) ** 2 / 2.0)
    return cost

"""
Now put all the fn together w/ a model training fn by doing the following

1) update the weight vector in each iteration

2) print out current cost per k iterations to ensure cost is decreasing 
"""

def train_linear_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """
    Train a linear regression model w/ gradient descent, and return trained model
    """

    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        # Check the cost for every 100 (or k) iterations 
        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

# finally predict the results of new input values using the trained model

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        x = np.hstack((intercept, X))
    return compute_prediction(X, weights)        

# a small example

X_train = np.array([[6], [2], [3], [4], [1], [5], [2], [6], [4], [7]])

y_train = np.array((5.5, 1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8))

weights = train_linear_regression(X_train, y_train, max_iter=100, learning_rate=0.01, fit_intercept=True)

# check model performance on new samples 

X_test = np.array([[1.3], [3.5], [5.2], [2.8]])

predictions = predict(X_test, weights)

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, marker='o', c='b')
plt.scatter(X_test[:, 0], predictions, marker='o', c='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

