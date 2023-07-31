"""
Exploring the clothing image dataset

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 388). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf 

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print some examples

print(train_labels)

# label arrays don't include class names. 
# so we define them here and use them for plotting later

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shift', 'Sneaker', 'Bag', 'Ankle boot']

# take a look at the format of the image data as follows

print(train_images.shape)   # 60k training samples

# similarly for the 10k testing samples, we check the format

print(test_images.shape)

# inspect the random training sample

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(train_images[42])
plt.colorbar()
plt.grid(False)
plt.title(class_names[train_labels[42]])
plt.show()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# in the ankle book sample, the pixel values are 0 to 255. 
# so we rescale the data to a range of 0 to 1 
# before feeding it to the neural network
# we divide the values of both training samples, test samples by 255

train_images = train_images / 255.0
test_images = test_images / 255.0

# display the first 16 training samples after the preprocessing

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=.3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[i]])
plt.show()

# first reshape the data

X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))

print(X_train.shape)

# specify the random seed in tf for reproducibility

tf.random.set_seed(42)

# we import the necessary modules from keras and
# initialize a keras-based model

from tensorflow import keras
from keras import datasets, layers, models, losses
model = models.Sequential()

# we are going to build 3 convolutional layers
# the first layer has 32 3x3 filters. 

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# note we use the relu as the activation function

# a max pooling layer w/ a 2x2 filter is next

model.add(layers.MaxPooling2D((2, 2)))

# here is the second convolutional layer
# it has 64 3x3 filters and comes w/ a relu activation function

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# the second convolutional layer is followed by 
# another max-pooling layer w/ a 2x2 filter

model.add(layers.MaxPool2D((2, 2)))




