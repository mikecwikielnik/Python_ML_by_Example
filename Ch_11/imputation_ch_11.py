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

# similarly, initialize the imputation transformer w/ the median value

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(data_origin)
data_median_imp = imp_median.transform(data_origin)
print(data_median_imp)

# when new samples come in, the missing values can be imputed 
# using the trained transformer.
# ex: the mean value

new = [[20, np.nan],
       [30, np.nan],
       [np.nan, 70],
       [np.nan, np.nan]]
new_mean_imp = imp_mean.transform(new)
print(new_mean_imp)

# ex: the strategy of imputing missing values and discarding missing data
# affects the prediction results

# first, load diabetes dataset

from sklearn import datasets

dataset = datasets.load_diabetes()
X_full, y = dataset.data, dataset.target    # important! 

