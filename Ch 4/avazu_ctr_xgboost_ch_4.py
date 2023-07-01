"""
Ensembling decision trees - gradient boosted trees

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 142). Packt Publishing. Kindle Edition. 
"""

import pandas as pd
n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# first, we transform the label variable into 2 dim. 
# 0 -> [1,0] & 1 -> [0,1]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

# next. we train the GBT model on the training set we prepared previously:

import xgboost as xgb
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 10, n_estimators = 1000)

model.fit(X_train_enc, Y_train)

# we use the trained model to make predictions on the testing set and calculate ROC AUC accordingly

pos_prob = model.predict_proba(X_test_enc)[:, 1]

from sklearn.metrics import roc_auc_score
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')


