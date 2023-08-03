"""
Writing your own War and Peace with RNNs

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 431). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

import numpy as np

# first we read the file and convert the text into lowercase

training_file = 'warpeace_input.txt'

raw_text = open(training_file, 'r').read()
raw_text = raw_text.lower()

# we take a quick look at the training text data by printing out the first 200 characters

print(raw_text[:200])

# next we count the number of unique words

all_words = raw_text.split()
unique_words = list(set(all_words))
print(f'Number of unique words: {len(unique_words)}')

# then we cound the total number of characters

n_chars = len(raw_text)
print(f'Total characters: {n_chars}')

# from the 3 million characters, we obtain the unique characters

chars = sorted(list(set(raw_text)))
n_vocab = len(chars)
print(f'total vocabulary (unique characters): {n_vocab}')
print(chars)

# we need to one-hot encode the input and output characters since NN models
# only take in numerical data

index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))
print(char_to_index)

# now that the character lookup dictionary is ready, we can construct the entire training set

seq_length = 160
n_seq = int(n_chars / seq_length)

