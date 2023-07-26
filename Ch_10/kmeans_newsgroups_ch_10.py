"""
Clustering newsgroups data using k-means

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 333). Packt Publishing. Kindle Edition. 
"""

from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space'
]

groups = fetch_20newsgroups(subset='all', categories=categories)

labels = groups.target
label_names = groups.target_names

from nltk.corpus import names
all_names = set(names.words())

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)

# we then convert the cleaned text data into count vectors using CountVectorizer of sklearn

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

data = count_vector.fit_transform(data_cleaned)

from sklearn.cluster import KMeans

k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)

clusters = kmeans.labels_

from collections import Counter
print(Counter(clusters))

# using tf-idf representation, we replace CountVectorizer with TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

