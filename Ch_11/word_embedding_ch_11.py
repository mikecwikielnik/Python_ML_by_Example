"""
Word embedding with pre-trained models

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 365). Packt Publishing. Kindle Edition. 
"""

import gensim.downloader as api

model = api.load("glove-twitter-25")