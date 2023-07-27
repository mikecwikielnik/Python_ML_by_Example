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

# select different number of top features and 
# estimate the accuracy on each dataset

K = [10, 15, 25, 35, 45]
for k in K:
    top_k_features = feature_sorted[-k:]
    X_k_selected = X[:, top_k_features]
    # Estimate accuracy on the data set w/ k selected features
    classifier = SVC(gamma=0.005)
    score_k_features = cross_val_score(classifier, X_k_selected, y).mean()
    print(f'Score with the dataset of top {k} features: {score_k_features:.2f}')

    