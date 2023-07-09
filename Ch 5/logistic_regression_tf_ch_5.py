"""
Implementing logistic regression using TensorFlow

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 178). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf
import pandas as pd
n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train].astype('float32')
X_test = X[n_train:]
Y_test = Y[n_train:].astype('float32')

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train).toarray().astype('float32')
X_test_enc = enc.transform(X_test).toarray().astype('float32')

# we use the tf.data api to shuffle & batch data

batch_size = 1000
train_data = tf.data.Dataset.from_tensor_slices((X_train_enc, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# we define the weights and bias of the logistic regression model:

n_features = int(X_train_enc.shape[1])
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))

# we then create a gradient descent optimizer that searches for the best coef by minimizing the loss
# we herein use Adam as our optimizer

learning_rate = 0.0008
optimzier = tf.optimizers.Adam(learning_rate)