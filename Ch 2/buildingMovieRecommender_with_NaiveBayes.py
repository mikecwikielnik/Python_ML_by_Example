"""
CH 2
Building a Movie Recommendation Engine with Naïve Bayes

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 43). Packt Publishing. Kindle Edition. 
"""

# Building a movie recommender with Naïve Bayes

# Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 60). Packt Publishing. Kindle Edition. 

# Let's attempt to determine whether a user likes a particular movie based on how users rate other movies (ratings are from 1 to 5).

# First, we import all the necessary modules and variables:

import numpy as np
from collections import defaultdict
data_path = 'C:/Users/mikec/OneDrive/Google One Drive/Google Drive/Py_files/Python/Python ML by Example, 3rd Ed/Python-Machine-Learning-By-Example-Third-Edition-master/chapter2/m1-1m/ratings.dat'
n_users = 610
n_movies = 9742

# We then develop the following function to load the rating data from ratings.dat:

def load_rating_data(data_path, n_users, n_movies):
    """
    Load rating data from file and also return the number of ratings
        for each movie and movie_id index mapping
    @param data_path: path of the rating data file
    @param n_users: number of users
    @param n_movies: number of movies that have ratings
    @return: rating data in the numpy array of [user, movie];
        movie_n_rating, {movie_id: number of ratings};
        movie_id_mapping, {movie_id: column index in rating data}
    """

    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split(",")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(float(rating))
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    
    return data, movie_n_rating, movie_id_mapping

# And then we load the data using this function:

data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)



# It is always recommended to analyze the data distribution. We do the following:

def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')

display_distribution(data)


# since most ratings are unknown, we take the movie with the most known ratings as our target movie:

movie_id_most, n_rating_most = sorted(movie_n_rating.items(), 
                                      key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')

# the movie yielded is our target movie, and ratings of the rest are signals. We construct the dataset accordingly:

X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

# we discard samples without a rating in the movie yielded:

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

# again, we take a look at the distribution of the target movie ratings:

display_distribution(Y)

# we consider movies with ratings greater than 3 as being liked (being recommended):

recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples.')

# as a rule of thumb in solving classification problems, 
# we need to always analyze the label classification and see how balanced (or imbalanced) the dataset is.

# we use the 'train_test_split' fn from scikit-learn to do the random splitting and 
# to preserve the percentage of samples for each class:

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# we check the training and testing sizes as follows:

print(len(Y_train), len(Y_test))

# we import MultionomialNB , initialize a model with a smoothing factor of 1.0 and prior learned from the training set,
# and train this model against the training set as follows:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

# then, we use the trained model to make predictions on the testing set. 
# We get the predicted probabilities as follows:

prediction_prob = clf.predict_proba(X_test)
prediction_prob[0:10]

# we get the predicted class as follows:

prediction = clf.predict(X_test)
print(prediction[:10])

# finally, we evaluate the model's performance with classification accuracy, which is the proportion of correct predictions:

accuracy = clf.score(X_test, Y_test)
print(f"The accuracy is: {accuracy*100:.1f}%")