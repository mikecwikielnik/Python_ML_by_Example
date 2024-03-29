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

# to visualize this, execute the following code:

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c=['b'] * 5 + ['k'] * 5, marker = 'o')

# blue dots are training samples from class 0, while black dots are those from class 1.
# use 0.5 as the classification decision threshold:

colours = ['k' if prediction >= 0.5 else 'b' for prediction in predictions]
plt.scatter(X_test[:, 0], X_test[:, 1], marker='*', c=colours)

# blue stars are testing samples predicted from class 0, black stars are predicted from class 1

plt.xlabel('x1')
plt.xlabel('x2')
plt.show()

# Predicting ad click-through with logistic regression using gradient descent

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 165). Packt Publishing. Kindle Edition. 

"""
After a brief example, we will now deploy the algo we just developed in
our click-through prediction project
"""

import pandas as pd

n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = 100000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)

X_test_enc = enc.transform(X_test)

# train a logistic regression model over 10000 iterations, at a learning rate of 0.01 with bias:

import timeit
start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_enc.toarray(), Y_train, max_iter = 10000, learning_rate = 0.01,
                                    fit_intercept = True)
print(f"---{(timeit.default_timer() - start_time)}.3fs seconds ---")

# the trained model performs on the testing set as follows:

pred = predict(X_test_enc.toarray(), weights)
from sklearn.metrics import roc_auc_score
print(f'Training samples: {n_train}, AUC on testing set: {roc_auc_score(Y_test, pred):.3f}')

# to implement SGD-based log regression, we slightly modify the update_weights_gd function:

def update_weights_sgd(X_train, y_train, weights, learning_rate):
    """ One weight update iteration: moving weights by one step based on each individual sample
    Args:
        X_train, y_train (numpy.ndarry, training data set)
        weights (numpy.ndarray)
        learning_rate (float)
    Returns:
        numpy.ndarray, updated weights
    """
    for X_each, y_each in zip(X_train, y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (y_each - prediction)
        weights += learning_rate * weights_delta
    return weights

# in the train_logistic_regression function, SGD is applied:

def train_logistic_regression_sgd(X_train, y_train, max_iter, learning_rate, fit_intercept = False):
    """ Train a logistic regression model via SGD
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        max_iter (int, number of iterations)
        learning_rate (float)
        fit_intercept (bool, with an intercept w0 or not)
    Returns:
        numpy.ndarray, learned weights
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_sgd(X_train, y_train, weights, learning_rate)
        # Check the cost for every 2 (for example) iterations
        if iteration % 2 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

# train the sgd model based on 100000 samples
start_time = timeit.default_timer()
weights = train_logistic_regression_sgd(X_train_enc.toarray(), Y_train, max_iter=10, learning_rate=0.01, 
                                        fit_intercept=True)
print(f"---{(timeit.default_timer() - start_time)}.3fs seconds ---")                        
pred = predict(X_test_enc.toarray(), weights)
print(f'Training samples: {n_train}, AUC on testing set: {roc_auc_score(Y_test, pred):.3f}')

