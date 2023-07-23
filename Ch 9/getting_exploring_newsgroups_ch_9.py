"""
Getting the newsgroups data

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 294). Packt Publishing. Kindle Edition. 
"""

from sklearn.datasets import fetch_20newsgroups

# downlod the dataset w/ all the default parameters

groups = fetch_20newsgroups()
groups.keys()

# target_names key gives the newsgroups names

groups['target_names']

# target key corresponds to a newsgroup but encoded as an integer

groups.target

# what are the distinct values for these integers?
# we use the unique fn from numpy to figure it out

import numpy as np
np.unique(groups.target)

# 0-19 means the 1st, 2nd, etc newsgroup topics in 
# groups['target_names']

# plot the distribution of the newsgroup topics

import seaborn as sns
import matplotlib.pylab as plt

sns.distplot(groups.target)
plt.show()

