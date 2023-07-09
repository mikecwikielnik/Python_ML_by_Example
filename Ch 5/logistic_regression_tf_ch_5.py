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
