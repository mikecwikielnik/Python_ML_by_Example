"""
Implementing k-means with scikit-learn

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 329). Packt Publishing. Kindle Edition. 
"""

# first, import the Kmeans class & initialize a model w/ 3 clusters

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

from matplotlib import pyplot as plt

k = 3
from sklearn.cluster import KMeans

Kmeans_sk = KMeans(n_clusters=3, random_state=42)

# we then fit the model on the data
  
Kmeans_sk.fit(X)

# after fitting the model, we can get the clustering results
# including clusters for data samples and centroids of individual clusters

clusters_sk = Kmeans_sk.labels_
centroids_sk = Kmeans_sk.cluster_centers_

# similarly, we plot the clusters along with the centroids

plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()


