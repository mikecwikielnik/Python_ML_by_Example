"""
Boosting the CNN classifier with data augmentation

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 400). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['t-shirt/top', 'trouser','pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

