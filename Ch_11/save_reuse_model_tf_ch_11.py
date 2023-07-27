"""
Saving and restoring models in TensorFlow

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 376). Packt Publishing. Kindle Edition. 
"""

"""
we will train a simple logistic regression model on the cancer dataset, save the trained model,
and reload it in the following steps:
"""

# 1) import the necessary TF modules and load the cancer dataset from sklearn

import tensorflow as tf
from tensorflow import keras

from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
X = cancer_data.data    
Y = cancer_data.target

# 2) build a simple logistic regression model w/ keras sequential api & several parameters

learning_rate = 0.005
n_iter = 10

tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate))

              