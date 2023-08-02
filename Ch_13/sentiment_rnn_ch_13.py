"""
Analyzing movie review sentiment with RNNs

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 423). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras import layers, models, losses, optimizers
from keras.preprocessing.sequence import pad_sequences

# keras has a bulit in imdb dataset, so load the dataset

vocab_size = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

# take a look at the training, testing data

print('Number of training samples: ', len(y_train))
print('Number of positive samples', sum(y_train))

print('Number of test samples:', len(y_test))

# print a training sample

print(X_train[0])

# we use the word dictionary to map the integer back to the word it represents

word_index = imdb.get_word_index()
index_word = {index: word for word, index in word_index.items()}