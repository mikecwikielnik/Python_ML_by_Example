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

# loop over every 100000 samples & partially fit the model:
import timeit
start_time = timeit.default_timer()

for i in range(10):
    x_train = X_train[i*100000: (i+1)*100000]
    y_train = Y_train[i*100000: (i+1)*100000]
    x_train_enc = enc.transform(x_train)
    sgd_lr_online.partial_fit(x_train_enc.toarray(), y_train, classes=[0, 1])

print(f"--- {(timeit.default_timer() - start_timme)}.3fs seconds ---")

# apply the trained model on the testing set, the next 100k samples, as follows:

x_test_enc = enc.transform(X_test)

pred = sgd_lr_online.predict_proba(x_test_enc.toarray())[:, 1]
print(f'Training samples: {n_train * 10}, AUC on testing set: {roc_auc_score(Y_test, pred)}')

