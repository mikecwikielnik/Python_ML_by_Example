# Chapter 4
# Predicting Online Ad Click-Through with Tree-Based Algorithms

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 109). Packt Publishing. Kindle Edition. 

import matplotlib.pyplot as plt
import numpy as np

# plot gini impurity in a binary case

pos_fraction = np.linspace(0.00, 1.00, 1000)
gini = 1 - pos_fraction**2 - (1-pos_fraction)**2

plt.plot(pos_fraction, gini)
plt.ylim(0, 1)
plt.xlabel('Positive Fraction')
plt.ylabel('Gini Impurity')
plt.show()

def gini_impurity(labels):
    # when the set is empty, it is also pure
    if not labels:
        return 0
    # count the occurences of each label
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)

print(f'{gini_impurity([1, 1, 0, 1, 0]):.4f}')
print(f'{gini_impurity([1, 1, 0, 1, 0, 0]):.4f}')
print(f'{gini_impurity([1, 1, 1, 1]):.4f}')

# visualizing entropy in a binary case

pos_fraction = np.linspace(0.00, 1.00, 1000)
ent = -(pos_fraction * np.log2(pos_fraction) + 
        (1 - pos_fraction) * np.log2(1 - pos_fraction))
plt.plot(pos_fraction, ent)
plt.xlabel('positive fraction')
plt.ylabel('entropy')
plt.ylim(0, 1)
plt.show()

# given the labels of a dataset, the entropy calculation function

def entropy (labels):
    if not labels:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return - np.sum(fractions * np.log2(fractions))

print(f'{entropy([1, 1, 0, 1, 0]):.4f}')
print(f'{entropy([1, 1, 0, 1, 0, 0]):.4f}')
print(f'{entropy([1, 1, 1, 1]):.4f}')

# combine gini impurity and information gain (entropy-based evaluation) to calc the weighted impurity

criterion_function = {'gini': gini_impurity, 'entropy': entropy}

def weighted_impurity(groups, criterion='gini'):
    """
    Calculate weighted impurity of children after a split
    @param groups: list of children, and a child consists a list of class labels
    @param criterion: metric to measure the quality of a split, 
        'gini' for Gini Impurity or 'entropy' for Information Gain
    @return: float, weighted impurity
    """

    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum

children_1 = [[1, 0, 1], [0, 1]]
children_2 = [[1, 1], [0, 0, 1]]

print(f"Entropy of #1 split: {weighted_impurity(children_1, 'entropy'):.4f}")
print(f"Entropy of #2 split: {weighted_impurity(children_2, 'entropy'):.4f}")