"""
Feature selection using random forest

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 180). Packt Publishing. Kindle Edition. 
"""

import numpy as np
import pandas as pd
n_rows = 100000
df = pd.read_csv("train", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values


X_train = X
Y_train = Y

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)

# we will examine feature selection w/ random forest on the dataset
# with 100k ad click samples

# feature selecttion with random forest

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
random_forest.fit(X_train_enc.toarray(), Y_train)

# after fitting the random forest model, we obtain the feature importance scores

feature_imp = random_forest.feature_importances_
print(feature_imp)

# bottom 10 weights & corresponding 10 least important features
feature_names = enc.get_feature_names_out()
print(np.sort(feature_imp)[:10])
bottom_10 = np.argsort(feature_imp)[:10]
print('10 least important features are:\n', feature_names[bottom_10])
