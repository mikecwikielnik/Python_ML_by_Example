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
        weighted_sum += len(group) / float(total) * mse(group)
    return weighted_sum

# test things out by executing the following commands:

print(f'{mse(np.array([1, 2, 3])):.4f}')
print(f'{weighted_mse([np.array([1, 2, 3]), np.array([1, 2])]):.4f}')

# to build the house price regression tree, we first exhaust all possible pairs of feature and value, & 
# we compute the corresponding mse

print(f'type-semi: {weighted_mse([np.array([600, 400, 700]), np.array([700, 800])]):.4f}')
print(f'bedroom-2: {weighted_mse([np.array([700, 400]), np.array([600, 800, 700])]):.4f}')
print(f'bedroom-3: {weighted_mse([np.array([600, 800]), np.array([700, 400, 700])]):.4f}')
print(f'bedroom-4: {weighted_mse([np.array([700]), np.array([600, 700, 800, 400])]):.4f}')

