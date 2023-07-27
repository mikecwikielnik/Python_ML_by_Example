"""
Best practice 8 – Deciding whether to select features, and if so, how to do so

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 356). Packt Publishing. Kindle Edition. 
"""

# first, we load the handwritten digits dataset from sklearn

import numpy as np
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target     # important
print(X.shape)

