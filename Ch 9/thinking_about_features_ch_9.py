"""
Thinking about features for text data

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 301). Packt Publishing. Kindle Edition. 
"""

from sklearn.datasets import fetch_20newsgroups

groups = fetch_20newsgroups()

from sklearn.feature_extraction.text import CountVectorizer

# first, initialize the count vectorizer w/ 500 top features (500 most frequent tokens):

count_vector = CountVectorizer(stop_words="english", max_features=500)

# use it to fit on the raw text data as follows:

data_count = count_vector.fit_transform(groups.data)

data_count
data_count[0]

print(count_vector.get_feature_names_out())

# ------ text processing ----------

# we retin letter-only words so numbers and letter/number combos are removed

data_cleaned = []
for doc in groups.data:
    doc_cleaned = ' '.join(word for word in doc.split() if word.isalpha())
    data_cleaned.append(doc_cleaned)

# ex: dropping stop words  

from sklearn.feature_extraction import _stop_words
print(_stop_words.ENGLISH_STOP_WORDS)

count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

# ex: putting preprocessing, dropping new words, lemmatizing, and count vectorizing together

from nltk.corpus import names
all_names = set(names.words())

count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)

data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)    

# now the features are much more meaningful

print(count_vector_sw.get_feature_names_out())