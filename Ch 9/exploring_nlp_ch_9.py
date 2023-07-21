"""
Touring popular NLP libraries and picking up NLP basics

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 285). Packt Publishing. Kindle Edition. 
"""

import nltk
nltk.download()

# first, import the names corpus (dataset)

from nltk.corpus import names

# we check out the first 10 names in the list:

print(names.words()[:10])