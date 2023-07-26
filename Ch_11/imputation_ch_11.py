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

# simulate a corrupted dataset by adding 25% missing values

m, n = X_full.shape
m_missing = int(m * 0.25)
print(m, m_missing)

# randomly select the m_missing samples

np.random.seed(42)
missing_samples = np.array([True] * m_missing + [False] * (m - m_missing))
np.random.shuffle(missing_samples)

# for each missing sample, randomly select 1 out of n features

missing_samples = np.random.randint(low=0, high=n, size=m_missing)

# represent missing values by nan

X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_samples] = np.nan

# discard samples containing missing values

X_rm_missing = X_missing[~missing_samples, :]
y_rm_missing = y[~missing_samples]

# estimate the R^2 w/ a regression forest model in a cross-validation manner
# estimate R^2 on the dataset w/ the missing samples removed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
regressor = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
score_rm_missing = cross_val_score(regressor, X_rm_missing, y_rm_missing).mean()
print(f'Score with the data set w/ missing samples removed: {score_rm_missing:.2f}')

# imputing the missing values w/ the mean

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_mean_imp = imp_mean.fit_transform(X_missing)