"""
Chapter 7

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 234). Packt Publishing. Kindle Edition. 
"""

# a small example of house price estimation 
# using the features house type and number of bedrooms

import numpy as np

# we define the MSE and weighted MSE computation functions 
# mean squared error calculation fn given continuous targets of a data set,

def mse(targets):
    # when the set is empty
    if targets.size == 0:
        return 0
    return np.var(targets)

# then we define the weighted MSE after a split in a node

def weighted_mse(groups):
    """
    calculate weighted MSE of children after a split
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_mse += len(group) / float(total) * mse(group)
    return weighted_mse

