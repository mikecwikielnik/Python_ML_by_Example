"""
CH 2
Building a Movie Recommendation Engine with Naïve Bayes

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 43). Packt Publishing. Kindle Edition. 
"""

# Implementing Naïve Bayes with scikit-learn

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 59). Packt Publishing. Kindle Edition. 

from sklearn.naive_bayes import BernoulliNB

# From implementingNaiveBayes_from_scratch.py
import numpy as np

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]])

Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])

# Let's intialiaze a model with a smoothing factor (specified as alpha in scikit-learn) of 1.0
# And prior learned from the training set (specified as fit_prior=True in scikit-learn)

clf = BernoulliNB(alpha=1.0, fit_prior=True)

# To train the Naive Bayes classifier with the fit method, we use the following line of code:

clf.fit(X_train,Y_train)

# And to obtain the predicted probability results with the predict_proba method, we use the following lines of code:

pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted Probabilities:\n', pred_prob)

# Finally, we do the following to directly acquire the predicted class with the predicted class with the predict method
# (0.5 is the default threshold, and if the predicted probability of class Y is greater than 0.5, class Y is assigned; otherwise N is used):

pred = clf.predict(X_test)
print('[scikit-learn] Prediction:', pred)

# The prediction results using scikit-learn are consistent with what we got using our own solution. 
# Now that we've implemented the algorithm both from scratch and using scikit-learn, why don't we use it to solve the movie recommendation problem?


