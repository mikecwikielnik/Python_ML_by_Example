# Implementing SVM

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 84). Packt Publishing. Kindle Edition. 


# 1) first load the dataset and do some basic analysis, as follows:

from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

X = cancer_data.data
Y = cancer_data.target

print('Input data size:', X.shape)

print('Output data size:', Y.shape)

print('Label names:', cancer_data.target_names)

n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()

print(f'{n_pos} positive samples and {n_neg} negative samples.')

# 2) we split the data into training and testing sets:

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# 3) first, we initialize a SVC model with the kernel set to linear, and 
# the penalty hyperparameter C set to the default of 1.0

from sklearn.svm import SVC 
clf = SVC(kernel='linear', C=1.0, random_state=42)

# 4) we then fit our model on the training data as follows:

clf.fit(X_train, Y_train)

# 5) and we predict on the testing set wtih the trained model and 
# obtain the prediction accuracy directly:

accuracy = clf.score(X_test, Y_test)

print(f"The accuracy is: {accuracy*100:.1f}%")