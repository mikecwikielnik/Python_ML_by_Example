"""
SGDClassifier module of scikit-learn:

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 168). Packt Publishing. Kindle Edition. 
"""
import numpy as np
from sklearn.metrics import roc_auc_score


import pandas as pd
n_rows = 300000
df = pd.read_csv("train", nrows=n_rows)

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

# # use scikit-learn package
from sklearn.linear_model import SGDClassifier
sgd_lr = SGDClassifier(loss = 'log', penalty=None, fit_intercept=True, max_iter=10, learning_rate='constant', eta0=0.01)

# train the model and test it

sgd_lr.fit(X_train_enc.toarray(), Y_train)
pred = sgd_lr.predict_proba(X_test_enc.toarray())[:, 1]
print(f'Training samples: {n_train}, AUC on testing set: {roc_auc_score(Y_test, pred):.3f}')

# ---------------------------------------
# feature selection with L1 regularization
sgd_lr_l1 = SGDClassifier(loss='log', penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=10, learning_rate='constant', eta0=0.01)
sgd_lr_l1.fit(X_train_enc.toarray(), Y_train)

# with the trained model, we obtain the absolute values of its coef
coef_abs = np.abs(sgd_lr_l1.coef_)
print(coef_abs)

# bottom 10 weights and the corresponding 10 least important features
print(np.sort(coef_abs)[0][:10])

feature_names = enc.get_feature_names()
bottom_10 = np.argsort(coef_abs)[0][:10]
print('10 least important features are:\n', feature_names[bottom_10])

# top 10 weights and the corresponding 10 most important features 
print(np.sort(coef_abs)[0][:10])
top_10 = np.argsort(coef_abs)[0][:10]
print('10 most important features are:\n', feature_names[top_10])

# ---------------------------------------------
# online learning

n_rows = 100000 * 11
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = 100000 * 10
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# fit the encoder on the whole training set as follows:

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)

# the number of iterations is set to 1 if using partial_fit
sgd_lr_online = SGDClassifier(loss='log', penalty=None, fit_intercept=True, max_iter=1, learning_rate='constant', eta0=0.01)

