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