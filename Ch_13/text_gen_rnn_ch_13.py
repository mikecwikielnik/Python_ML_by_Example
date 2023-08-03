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

# next we initialize the training inputs, outputs
# both are the shape (number of samples, sequence length, feature dimension)

X = np.zeros((n_seq, seq_length, n_vocab))
Y = np.zeros((n_seq, seq_length, n_vocab))

# now for each of the n_seq samples, we assing "1" to the indices of the input vectors
# and output vectors where the corresponding characters exist

for i in range(n_seq):
    x_sequence = raw_text[i * seq_length : (i + 1) * seq_length]
    x_sequence_ohe = np.zeros((seq_length, n_vocab))
    for j in range(seq_length):
        char = x_sequence[j]
        index = char_to_index[char]
        x_sequence_ohe[j][index] = 1
    X[i] = x_sequence_ohe
    y_sequence = raw_text[i * seq_length + 1: (i + 1) * seq_length + 1]
    y_sequence_ohe = np.zeros((seq_length, n_vocab))
    for j in range(seq_length):
        char = y_sequence[j]
        index = char_to_index[char]
        y_sequence_ohe[j][index] = 1 
    Y[i] = y_sequence_ohe

# now take a look at the shapes of the constructed input, output samples

print(X.shape)
print(Y.shape)

# training set is ready and it is time to build, fit RNN model

# ----------------- Building an RNN text generator ------------------

# we wil build a RNN w/ two stacked recurrent layers

# fix a random seed

tf.random.set_seed(42)

# each recurrent layer contains 700 units, w/ a 0.4 dropout ratio and tanh activation fn

hidden_units = 700
dropout = 0.4

# we specify other hyperparameters including batch size, and number of epochs

batch_size = 100
n_epoch = 300

# we create the RNN as follows

model = models.Sequential()
model.add(layers.LSTM(hidden_units, input_shape=(None, n_vocab), return_sequences=True, dropout=dropout))
model.add(layers.LSTM(hidden_units, return_sequences=True, dropout=dropout))

model.add(layers.TimeDistributed(layers.Dense(n_vocab, activation='softmax')))

# next we compile the network. We choose RMSprop w/ a learning rate of 0.001

optimizer = optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# lets summarize the model 

print(model.summary())

# ----------------- training the RNN text generator -------------------

filepath = "weights/weights_epoch_{epoch:03d}_loss_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

# we create an early stopping callback to halt the training if the validation loss doesn't decrease for 50 successive epochs

early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=1, mode='min')

# we develop a helper function that generates text of any length, given a model

def generate_text(model, gen_length, n_vocab, index_to_char):
    """
    generating text using the RNN model
    @param model: current RNN model
    @param gen_length: number of characters we want to generate
    @param n_vocab: number of unique characters
    @param index_to_char: index to character mapping
    @return: 
    """

    # start w/ a randomly picked character
    index = np.random.randint(n_vocab)
    y_char = [index_to_char[index]]
    X = np.zeros((1, gen_length, n_vocab))
    for i in range(gen_length):
        X[0, i, index] = 1.
        indices = np.argmax(model.predict(X[:, max(0, i - 99):i + 1, :])[0], 1)
        index = indices[-1]
        y_char.append(index_to_char[index])
    return ''.join(y_char)

# now we define the callback class that generates text with the generate_text fn for every n epochs

class ResultCheck(Callback):
    def __init__(self, model, N, gen_length):
        self.model = model
        self.N = N
        self.gen_length = gen_length

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.N == 0:
            result = generate_text(self.model, self.gen_length, n_vocab, index_to_char)
            print('\nMy War and Peace:\n' + result)

            