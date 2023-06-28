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

