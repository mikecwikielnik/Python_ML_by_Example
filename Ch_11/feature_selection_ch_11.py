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

# next, estimate the accuracy of the original dataset,
# which is 64-dimensional

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(gamma = 0.005, random_state=42)
score = cross_val_score(classifier, X, y).mean()
print(f'Score with the original data set: {score:.2f}')

