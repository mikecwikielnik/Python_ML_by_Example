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

# take the first review as an example

print([index_word.get(i, ' ') for i in X_train[0]])

# we analyze the length of each sample. We do so because all the input sentences
# to a RNN model must be the same length

review_lengths = [len(x) for x in X_train]

# plot the distribution of these document lengths

from matplotlib import pyplot as plt
plt.hist(review_lengths, bins=10)
plt.show()

# the visualization shows that most of the reviews are 200 words long
# next, we set 200 as the universal sequence length by padding shorter reviews, truncating longer reviews
# we use the pad_sequences fn from keras to do this

maxlen = 200

X_train = pad_sequences(X_test, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# lets look at the shape of the input sequence after this

print('X_train shape after padding:', X_train.shape)
print('X_test shape after padding:', X_test.shape)

# ---- building a simple LSTM network --------------------

# now that the training, testing datasets are ready, we can build our first RNN model

# first, we fix the random seed and initiate a keras sequential model

tf.random.set_seed(42)

model = models.Sequential()

# since our input sequences are word indices that are equivalent to one-hot encoded vectors
# we need to embed them in dense vectors using the embedding layer from keras

embedding_size = 32

model.add(layers.Embedding(vocab_size, embedding_size))

# now here comes the recurrent layer, the LSTM layer specifically

model.add(layers.LSTM(50))

# herein, we only use one recurrent layer w/ 50 nodes

# after, we add the output layer, along with the sigmoid activation function, 
# since we are working on a binary classification problem

model.add(layers.Dense(1, activation='sigmoid'))

