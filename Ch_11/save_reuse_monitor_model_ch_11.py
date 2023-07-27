"""
Saving and restoring models using pickle

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 374). Packt Publishing. Kindle Edition. 
"""

from sklearn import datasets
dataset = datasets.load_diabetes()
X, y = dataset.data, dataset.target

num_new = 30    # the last 30 samples as new data set 
X_train = X[:-num_new, ::]
y_train = y[:-num_new]
X_new = X[-num_new:, :]
y_new = y[-num_new:]

# data preprocessing with scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# now save the established standardizer

import pickle
# save the scaler
pickle.dump(scaler, open("scaler.p", "wb"))

# move on to training an SVR model on the scaled data

X_scaled_train = scaler.transform(X_train)

# regression model training

from sklearn.svm import SVR 

regressor = SVR(C=20)
regressor.fit(X_scaled_train, y_train)

# save the regressor 

pickle.dump(regressor, open("regressor.p", "wb"))

# deployment
# we first load the saved standardizer &
# the regressor object from the preceding 2 files

my_scaler = pickle.load(open("scaler.p", "rb"))
my_regressor = pickle.load(open("regressor.p", "rb"))

