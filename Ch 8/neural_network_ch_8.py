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

# define the training fn, which takes the training dataset, 
# the number of units in the hidden layer (we use 1 hidden layer as ex)
# and the number of iterations

def train(X, y, n_hidden, learning_rate, n_iter):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))
    for i in range(1, n_iter + 1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3

        dZ3 = A3 - y
        dW2 = np.matmul(A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
        dW1 = np.matmul(X.T, dZ2)
        db1 = np.sum(dZ2, axis=0)

        W2 = W2 - learning_rate * dW2 / m
        b2 = b2 - learning_rate * db2 / m
        W1 = W1 - learning_rate * dW1 / m
        b1 = b1 - learning_rate * db1 / m

        if i % 100 == 0:
            cost = np.mean((y - A3) ** 2)
            print('Iteration %i, training loss: %f' % (i, cost))

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

def predict(x, model):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    A2 = sigmoid(np.matmul(x, W1) + b1)
    A3 = np.matmul(A2, W2) + b2
    return A3

# define a prediction tn, which takes in a model & produce the regression results

# data normalization here. 
# we standardize the input data by removing the mean & scaling to unit variance

from sklearn import datasets
boston = datasets.load_boston()
num_test = 10 # the last 10 samples as testing set

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

X_train = boston.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = boston.target[:-num_test].reshape(-1, 1)
X_test = boston.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = boston.target[-num_test:]

# w/ the scaled dataset, we train a one-layer neural network 
# with 20 hidden units, a 0.1 learning ratae, 2k iterations

n_hidden = 20
learning_rate = 0.1
n_iter = 2000

model = train(X_train, y_train, n_hidden, learning_rate, n_iter)

# finally, we apply the trained model on the testing set

predictions = predict(X_test, model)
print(predictions)
print(y_test)

# we just made a neural network from scratch, time to do it with scikit-learn

# we utilize the MLPRegressor class (multi-layer perceptron, a nickname for neural networks)

from sklearn.neural_network import MLPRegressor

nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam',
                         learning_rate_init=0.001, random_state=42, max_iter=2000)

# hidden_layer_sizes represents the number of hidden neurons.
# two hidden layers w/ 16 & 8 nodes. ReLu activation is used

# we fit the neural network model on the training set and predict on the testing data

nn_scikit.fit(X_train, y_train)
predictions = nn_scikit.predict(X_test)
print(predictions)

# we calculate the MSE on the prediction

print(np.mean((y_test - predictions) ** 2))

# we just implemented a neural network w/ scikit-learn. 
# now lets do it with tensorflow

# tensorflow implementation of neural network

# first, we import the necessary modules, & set a random seed, which is recomended for reproducible modeling

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

# next we create a keras sequential model by passing 2 fully
# connected hidden layers w/ 20 nodes, 8 nodes respectively. Again, ReLu activation is used

model = keras.Sequential([
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1)
])

# and we compile the model by using Adam as the optimizer w/ 
# a learning rate of 0.02 and MSE as the learning goal:

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.02))

# the adam optimizer is a replacement for the stochastic gradient descent algo

# after defining the model, we now train it against the training set:

model.fit(X_train, y_train, epochs=300)

# we fit the model w/ 300 iterations. 

# finally, we use the trained model to predict the testing cases & 
# print out the predictions and their MSE

predictions = model.predict(X_test)[:, 0]
print(predictions)
print(np.mean((y_test - predictions) ** 2))

