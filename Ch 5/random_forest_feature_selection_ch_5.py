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
