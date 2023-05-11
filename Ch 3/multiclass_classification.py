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

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42)

# 3) we can now apply the svm classifier to the data. 
# We first initialize an SVC model and fit it against the training set:


from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)

# In an SVC model, multiclass support is implicitly handled according to the one vs one scheme

# 4) next, we predict  on the testing set with the trained model and obtain the prediction accuracy directly;

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy}%')

# 5) we also check how it performs for individual classes

from sklearn.metrics import classification_report

pred = clf.predict(X_test)
print(classification_report(Y_test, pred))