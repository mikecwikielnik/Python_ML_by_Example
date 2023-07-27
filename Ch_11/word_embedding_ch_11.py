"""
Word embedding with pre-trained models

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 365). Packt Publishing. Kindle Edition. 
"""

import gensim.downloader as api

model = api.load("glove-twitter-25")

# we get the embedding vector for a word, for example: computer

vector = model.wv['computer']
print('Word computer is embedded into:\n', vector)

# get top 10 words that are the most relevant to computer 
# using most_similar method

similar_words = model.most_similar("computer")
print('top ten words most relevant to computer:\n', similar_words)

