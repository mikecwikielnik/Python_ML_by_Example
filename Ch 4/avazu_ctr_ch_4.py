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

