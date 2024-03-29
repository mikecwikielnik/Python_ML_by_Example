"""
Elbow method,

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 332). Packt Publishing. Kindle Edition. 
"""

from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

# sse = sum of squared errors
# sse is computed as the sum of the squared distances
# from individual samples in the cluster TO the centroid

# ex: we perform k-means clusterting under different values of k on the iris data

iris = datasets.load_iris()
X = iris.data
y = iris.target

k_list = list(range(1, 7))
sse_list = [0] * len(k_list)

# we use the whole feature space and k ranges 1:6
# we then train individual models and record the resulting SSE

for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)

        sse += np.linalg.norm(X[cluster_i] - centroids[i])

    print('k={}, SSE={}'.format(k, sse))
    sse_list[k_ind] = sse

# finally, we plot the SSE versus the various k ranges

plt.plot(k_list, sse_list)
plt.show()


