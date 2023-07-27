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
# estimate accuracy on the original data set

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(gamma = 0.005, random_state=42)
score = cross_val_score(classifier, X, y).mean()
print(f'Score with the original data set: {score:.2f}')

# then conduct feature selection based on random forest and
# sort the features based on their importance
# feature selectioin w/ random forest

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=42)
random_forest.fit(X, y)

# sort features based on their importnacies

feature_sorted = np.argsort(random_forest.feature_importances_)