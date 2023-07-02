"""
Classifying data with logistic regression

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 151). Packt Publishing. Kindle Edition. 
"""

import numpy as np

def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

# input values are -8 to 8, & the output is as follows:

import matplotlib.pyplot as plt
z = np.linspace(-8, 8, 1000)
y = sigmoid(z)
plt.plot(z, y)
plt.axhline(y = 0, ls = 'dotted', color = 'k')
plt.axhline(y = 0.5, ls = 'dotted', color = 'k')
plt.axhline(y = 1, ls = 'dotted', color = 'k')
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel('z')
plt.ylabel('y(z)')
plt.show()

# plot sample cost vs y_hat (prediction), for y (truth) = 1

y_hat = np.linspace(0, 1, 1000)
cost = -np.log(y_hat)
plt.plot(y_hat, cost)
plt.xlabel('prediction')
plt.ylabel('cost')
plt.xlim(0, 1)
plt.ylim(0, 7)
plt.show()



