"""
Dealing with missing data

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 350). Packt Publishing. Kindle Edition. 
"""

import numpy as np
from sklearn.impute import SimpleImputer

# represent the unknown value by np.nan in numpy, as follows

data_origin = [[30, 100],
               [20, 50],
               [35, np.nan],
               [25, 80],
               [30, 70],
               [40, 60]]

# initialize the imputation transformer w/ the mean value & 
# get the mean value from the original data:

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data_origin)

# compute the missing values as follows

data_mean_imp = imp_mean.transform(data_origin)
print(data_mean_imp)
