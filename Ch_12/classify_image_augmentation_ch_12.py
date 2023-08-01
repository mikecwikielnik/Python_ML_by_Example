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
    for file in os.listdif(folder):
        if file.startswith(save_prefix):
            plt.subplot(2, 2, i + 1, xticks=[], yticks=[])
            aug_img = load_img(folder + "/" + file)
            plt.imshow(aug_img)
            plt.title(f"Augmented {i}")
            i += 1
    plt.show()

    