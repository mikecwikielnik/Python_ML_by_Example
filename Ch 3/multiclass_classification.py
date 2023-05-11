# Multiclass Classification

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 85). Packt Publishing. Kindle Edition. 

# A wine classification example with 3 classes

# 1) first load the data and do some basic analysis

from sklearn.datasets import load_wine  

wine_data = load_wine()

X = wine_data.data
Y = wine_data.target

print('Input size data: ', X.shape)
print('Output data size: ', Y.shape)
print('Label names: ', wine_data.target_names)

n_class0 = (Y == 0).sum()
n_class1 = (Y == 1).sum()
n_class2 = (Y == 2).sum()

print(f'{n_class0} class0 samples\n{n_class1} class1 samples\n{n_class2} class2 samples')

# 2) next, we split the data into training and testing sets

X_train, X_text, Y_train