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

