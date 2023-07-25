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

# now we perform step 3: assigning clusters based on the nearest centroids
# first, we need to define a fn calculating distance by the euclidean distance

def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

# then we develop a fn that assigns a sample to the cluster of the nearest centroid

def assign_cluster(x, centroids):
    distances = dist(x, centroids)  
    cluster = np.argmin(distances)
    return cluster

# step 4: update the centroids to the mean of all samples in the individual clusters

def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)

# step 5: involves repeating step 3, 4 until the model converges and whichever of the 
# following occurs:
# 1) centroids move less than the pre-specifid threshold
# 2) sufficient iterations have been taken

# we set the tolerance of the first condition and the max iterations

clusters = np.zeros(len(X))

tol = 0.0001
max_iter = 100



