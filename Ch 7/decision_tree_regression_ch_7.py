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

# the lowest mse is achieved w/ 'type, semi' pair,
# the root node is formed here by this splitting point.

# we can go further down by constructing the second level
# from the right branch (the left branch be split any more)

print(f'bedroom-2: {weighted_mse([np.array([]), np.array([600, 400, 700])]):.4f}')
print(f'bedroom-3: {weighted_mse([np.array([400]), np.array([600, 700])]):.4f}')
print(f'bedroom-4: {weighted_mse([np.array([400, 600]), np.array([700])]):.4f}')

# w/ the second splitting point specificied by 'bedroom, 3' (atleast 3 bedrooms or not)
# w/ the lowest mse. 

#-------------- implementing decision tree regression --------------------------

# the node splitting utility function is the same as chapter 5
# which separates samples in a node into left, right branches
# based on a feature and value pair

def split_node(X, y, index, value):
    """
    Split data set X, y based on a feature and a value
    @param index: index of the feature used for splitting
    @param value: value of the feature used for splitting
    @return: left and right child, a child is in the format of [X, y]
    """

    x_index = X[:, index]
    # if this feature is numerical
    if type(X[0, index]) in [int, float]:
        mask = x_index >= value
    # if the feature is categorical
    else:
        mask = x_index == value
    # split into left and right child
    left = [X[~mask, :], y[~mask]]
    right = [X[mask, :], y[mask]]
    return left, right

# next, we define the greedy search function
# trying out all possible splits & returning the one with the 
# least weighted MSE

def get_best_split(X, y):
    """
    Obtain the best splitting point and resulting children for the data set X, y
    @return: {index: index of the feature, value: feature value, children: left, right children}
    """

    best_index, best_value, best_score, children = None, None, 1e10, None
    
    for index in range(len(X[0])):
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, y, index, value)
            impurity = weighted_mse([groups[0][1], groups[1][1]])
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}

# when a stop criteria is met, the process at a node stops, and
# the mean value of the sample targets will be assigned to this terminal node:

def get_leaf(targets):
    # Obtain the leaft as the mean of the targets
    return np.mean(targets)


# split checks whether any stopping criteria are met and assigns the leaf node
# or proceeds w/ further separation otherwise

def split(node, max_depth, min_size, depth):
    """
    Split children of a node to construct new nodes or assign them terminals
    @param node: dict, with children info
    @param max_depth: maximal depth of the tree
    @param min_size: minimal samples required to further split a child
    @param depth: current depth of the node
    """

    left, right = node['children']
    del (node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    # check if the current depth exceeds the maximal depth
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    # check if the left child has enough samples
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        # it has enough samples, we further split it
        result = get_best_split(left[0], left[1])
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1)
    # check if the right child has enough samples
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        # it has enought samples, we further split it
        result = get_best_split(right[0], right[1])
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['right'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth + 1)

# the entry point of the regression tree construction is as follows:

def train_tree(X_train, y_train, max_depth, min_size):
    root = get_best_split(X_train, y_train)
    split(root, max_depth, min_size, 1)
    return root

# now, lets test it with a hand-calculated example

X_train = np.array([['semi', 3],
                    ['detached', 2],
                    ['detached', 3],
                    ['semi', 2],
                    ['semi', 4]], dtype=object)

y_train = np.array([600, 700, 800, 400, 700])

tree = train_tree(X_train, y_train, 2, 2)

# to verify the trained tree is identical to what we constructed by hand, 
# we write a function displaying the tree:

CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'is', 'no': 'is not'}}
def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if type(node['value']) in [int, float]:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print('{}[{}]'.format(depth * ' ', node))

# to verify the trained tree is identical to what we constructed by hand,
# we write a functino displaying the tree

tree = train_tree(X_train, y_train, 2, 2)
visualize_tree


# ---------------- boston house prices example -----------------

# directly use DecisionTreeRegressor from scikit-learn
from sklearn import datasets
boston = datasets.load_boston()

num_test = 10   # the last 10 samples as testing set

X_train = boston.data[:-num_test, :]
y_train = boston.target[:-num_test]
X_test = boston.data[-num_test:, :]
y_test = boston.target[-num_test:]

from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(max_depth = 10, min_samples_split=3)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)
print(y_test)

from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=3)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)

