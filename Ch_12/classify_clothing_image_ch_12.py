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

