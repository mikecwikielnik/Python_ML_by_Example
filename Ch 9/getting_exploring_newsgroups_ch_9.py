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