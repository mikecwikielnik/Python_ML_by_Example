"""
Thinking about features for text data

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 301). Packt Publishing. Kindle Edition. 
"""

# from sklearn.datasets import fetch_20newsgroups

# groups = fetch_20newsgroups()

from sklearn.feature_extraction.text import CountVectorizer

# first, initialize the count vectorizer w/ 500 top features (500 most frequent tokens):

count_vector = CountVectorizer(stop_words="english", max_features=500)

