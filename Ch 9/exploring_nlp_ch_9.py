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

# find out how many names there are

print(len(names.words()))

# a small tokenization function

from nltk.tokenize import word_tokenize

sent = """ I am reading a book.
            It is Python Machine Learning by Example,
            3rd edition."""
print(word_tokenize(sent))

# this example shows that tokenization is more complex than some think

sent2 = 'I have been to U.K. and U.S.A.'
print(word_tokenize(sent2))

