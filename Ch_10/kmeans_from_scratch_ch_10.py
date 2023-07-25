"""
Implementing k-means from scratch

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 317). Packt Publishing. Kindle Edition. 
"""

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target


# plot

import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# step 1: specifying k
# step 2: initializing centroids, by randomly selecting 3 samples as initial centroids

k = 3
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]

# visualize the data w.o labels any more, along w/ initial random centroids

def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()

visualize_centroids(X, centroids)
