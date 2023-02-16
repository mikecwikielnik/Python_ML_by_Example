"""
CH 2
Building a Movie Recommendation Engine with Naïve Bayes

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 43). Packt Publishing. Kindle Edition. 
"""

# Implementing Naïve Bayes

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 55). Packt Publishing. Kindle Edition. 

# After calculating by hand the movie preference, we are going to code Naive Bayes from scratch.
# After that, we will implement it using the scikit-learn package

# Implementing Naïve Bayes from scratch

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 56). Packt Publishing. Kindle Edition. 

# Before we develop the model,
# Lets define the toy dataset we just worked with:

import numpy as np

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]])

Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])

# For the model, starting with the prior,
# We first group the data by label and record their indices by classes:

def get_label_indices(labels):
    """
    Group samples based on their labels and return indices
    @param: list of lists
    @return: dict, {class1: [indices], class2: [indices]}
    """

    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

# Take a look at what we get:

label_indices = get_label_indices(Y_train)
print('label_indices:\n', label_indices)

# With label_indices, we calculate the prior:

def get_prior(label_indices):
    """
    Compute prior based on training sampmles
    @param label_indices: grouped sample indices by class
    @return: dictionary, with class label as key, corresponding 
        prior as the value
    """

    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())

    for label in prior:
        prior[label]/= total_count
    return prior

# Take a look at the computed prior:

prior = get_prior(label_indices)
print('Prior', prior)

# With prior calculated, we continue with likelihood,
# Which is the conditional probability, P(feature|class):

def get_likelihood(features, label_indices, smoothing=0):
    """
    Compute likelihood based on training samples
    @param features: matrix of features
    @param label_indices: grouped sample indices by class
    @param smoothing: integer, additive smoothing parameter
    @return: dictionary, with class as key, corresponding conditional probability 
        P(feature|class) vector as value.
    """

    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)

    return likelihood

# We set the smoothing value to 1 here, which can also be 0 for no smoothing,
# Or any other positive value, as long as a higher classification performance is achieved:

smoothing = 1
likelihood = get_likelihood(X_train, label_indices, smoothing)
print('Likelihood:\n', likelihood)

# Check figure 2.7 to refresh your memory

# With prior and likelihood ready, 
# We can now compute the posterior for the testing/new samples:

def get_posterior(X, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood
    @param X: testing samples
    @param prior: dictionary, with class label as key,
        corresponding prior as the value
    @param likelihood: dictionary, with class label as key,
        corresponding conditionaly probability vector as value
    @return: dictionary, with class label as key,
        corresponding posterior as value
    """

    posteriors = []
    for x in X:
        # posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
            
        # normalize so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        
        posteriors.append(posterior.copy())

    return posteriors

# Now, let's predict the class of our one sample test set using this prediction function:

posterior = get_posterior(X_test, prior, likelihood)
print('Posterior:\n', posterior)

# This is exactly what we got previously.
# We have successfully developed Naive Bayes from scratch and
# We can now move on to the implementation using scikit-learn.

