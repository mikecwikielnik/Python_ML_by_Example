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

# intialize the clusters starting value, along w/ all the starting clusters for all samples

iter = 0
centroids_diff = 100000

# w/ all the componenets ready, we can train the model by iteration

from copy import deepcopy
while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter += 1
    centroids_diff = np.linalg.norm(centroids - centroids_prev)
    print('Iteration:', str(iter))
    print('Centroids: \n', centroids)
    print('Centroids move: {:5.4f}'.format(centroids_diff))
    visualize_centroids(X, centroids)

    

