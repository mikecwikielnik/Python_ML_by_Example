"""
Best practice 12 – Performing feature engineering without domain expertise

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 360). Packt Publishing. Kindle Edition. 
"""

from sklearn.preprocessing import Binarizer

X = [[4], [1], [3], [0]]

binarizer = Binarizer(threshold=2.9)
X_new = binarizer.fit_transform(X)
print(X_new)

# polynomial transformation 

from sklearn.preprocessing import PolynomialFeatures

X = [[2, 4],
     [1, 3],
     [3, 2],
     [0, 3]]
poly = PolynomialFeatures(degree=2)
X_new = poly.fit_transform(X)
print(X_new)

