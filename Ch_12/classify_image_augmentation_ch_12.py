"""
Boosting the CNN classifier with data augmentation

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 400). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['t-shirt/top', 'trouser','pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# we will create manipulated images using this generator.
# first, we develop a utility fn to generate images
# given an augmented image generator and display them

# horizontal flipping

from keras.preprocessing.image import load_img

def generate_plot_pics(datagen, original_img, save_prefix):
    folder = 'aug_images'
    i = 0
    for batch in datagen.flow(original_img.reshape((-1, 28, 28, 1)),
                              batch_size=1,
                              save_to_dir = folder,
                              save_prefix = save_prefix,
                              save_format = 'jpeg'):
        i += 1
        if i > 2:
            break
    plt.subplot(2, 2, 1, xticks=[], yticks=[])
    plt.imshow(original_img)
    plt.title("Original")
    i = 1
    for file in os.listdir(folder):
        if file.startswith(save_prefix):
            plt.subplot(2, 2, i + 1, xticks=[], yticks=[])
            aug_img = load_img(folder + "/" + file)
            plt.imshow(aug_img)
            plt.title(f"Augmented {i}")
            i += 1
    plt.show()

# lets try out our horizontal_fli generator
# using the first training image

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True)
generate_plot_pics(datagen, train_images[0], 'horizontal_flip')

# horizontal & vertical flips simultaneously

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True)
generate_plot_pics(datagen, train_images[0], 'hv_flip')

# rotation for data augmentation

datagen = ImageDataGenerator(rotation_range=30)
generate_plot_pics(datagen, train_images[0], 'rotation')

# shifting for data augmentation

datagen = ImageDataGenerator(width_shift_range=8)
generate_plot_pics(datagen, train_images[0], 'width_shift')

# shifting both horizontally & vertically at the same time

datagen = ImageDataGenerator(width_shift_range=8,
                             height_shift_range=8)
generate_plot_pics(datagen, train_images[0], 'width_height_shift')

# improving the clothing image classifier w/ data augmentation

# since we can shift and flip images, we apply them to train our image classifier

# we start by constructing a small training set

X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))

n_small = 500
X_train = X_train[:n_small]
train_labels = train_labels[:n_small]

print(X_train.shape)

# we only use 500 samples for training

# we build the CNN model using the Keras Sequential API

from keras import datasets, layers, models, losses

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# we compile the model w/ Adam as the optimizer, 
# cross-entrophy as the loss fn,
# and classification accuracy as the metric:

model.compile(optimizer='adam',
              loss = losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])

# first we train the model w/o data aug

model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=20, batch_size=40)              

