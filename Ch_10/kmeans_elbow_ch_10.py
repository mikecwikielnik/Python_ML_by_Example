"""
Elbow method,

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 332). Packt Publishing. Kindle Edition. 
"""

from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

k_list = list(range(1, 7))
sse_list = [0] * len(k_list)

# sse = sum of squared errors
# sse is computed as the sum of the squared distances
# from individual samples in the cluster TO the centroid

# ex: we perform k-means clusterting under different values of k on the iris data

