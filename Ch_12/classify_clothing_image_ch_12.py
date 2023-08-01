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

# we continue adding the 3rd convolutional layer
# it has 128 3x3 filters 

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# resulting filter maps are flattened

model.add(layers.Flatten())

# for the classifier backend, we just use one hidden layer w/ 64 nodes

model.add(layers.Dense(64, activation='relu'))

# the output layer: has 10 nodes representing 10 different classes
# along with softmax activation

model.add(layers.Dense(10, activation='softmax'))

# now we compile the model w/ Adam as the optimizer, 
# cross-entropy as the loss function
# classification accuracy as the metric

model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# lets take a look at the model summary 

model.summary()


"""
now it's time to train the model we just built.
we train it for 10 iterations and evaluate it using the testing samples
"""

model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=10)

# accuracy is the training set- 0.95233
# val_accuracy is the testing set- 0.9119

# to double check the performance on the test set, 
# do this

test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
print('accuracy on test set:', test_acc)

# the above the check yield 0.91189

# now that we have a well-trained model,
# we can make predictions on the test set using the following

predictions = model.predict(X_test)

# take a look at the first sample, the prediction as follows

print(predictions[0])

