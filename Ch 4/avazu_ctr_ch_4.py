"""
avazu_ctr_ch_4.py

Predicting ad click-through with a decision tree

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 136). Packt Publishing. Kindle Edition. 
"""

import pandas as pd

n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)

print(df.head(5))

# the target variable is the click column:

Y = df['click'].values

# removing unnecessary featurs: id, hour, device_id, device_ip

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'],
            axis=1).values
print(X.shape)  # this literally basic R functions 

# if the data is in chronological order, just take the first 90% for the training set
# the balance for the testing set

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# discrete var into continuous var using OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

# we fit it on the training set as follows:

X_train_enc = enc.fit_transform(X_train)
X_train_enc[0]
print(X_train_enc[0])

# we transform the testing set using the trained one-hot encoder as follows:

X_test_enc = enc.transform(X_test)

# tweaking the max_depth hyperparameter

from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [3, 10, None]}

# initialize a decision tree model with Gini Impurity as the metric
# 30 as the min number of samples required to split further

decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
from sklearn.model_selection import GridSearchCV

# we use a 3-fold cross-validation and select the best performing
# hyperparameter measured by AUC:

grid_search = GridSearchCV(decision_tree, parameters, n_jobs= -1, cv= 3, scoring= 'roc_auc')

grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)

decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]

from sklearn.metrics import roc_auc_score
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')

import numpy as np
pos_prob = np.zeros(len(Y_test))
click_index = np.random.choice(len(Y_test), int(len(Y_test) * 51211.0/300000), replace=False)
pos_prob[click_index] = 1

print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')

# to employ a random forest, we use a package from scikit-learn

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs= -1)
grid_search = GridSearchCV(random_forest, parameters, n_jobs= -1, cv= 3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')

