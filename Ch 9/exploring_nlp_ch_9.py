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



# spaCy example

# we load en_core_web_sm model and parse the sentence using this model

import spacy
nlp = spacy.load('en_core_web_sm')
tokens2 = nlp(sent2)
print([token.text for token in tokens2])

from nltk.tokenize import sent_tokenize
print(sent_tokenize(sent))

# two sentences are returned, as there are two sentences as the input text

# PoS tagging example

import nltk

tokens = word_tokenize(sent)
print(nltk.pos_tag(tokens))
nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')

# each token parsed from an input sentence has
# an attribute called pos_, which is the tag we are looking for

print([(token.text, token.pos_) for token in tokens2])

# this was an example of PoS tagging with NLP packages

# NER example

# first, tokenize the input sentence

tokens3 = nlp('The book written by Hayden Liu in 2020 was sold at $30 in Americ.')

# the resulting token object contains an attribute called ents aka entities.
# we can extract the tagging for each entity

print([(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])

# ------ Stemming and lemmatization --------------

# import PorterStemmer, its a built-in stemming algo

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()



