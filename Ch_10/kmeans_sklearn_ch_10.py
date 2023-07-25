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