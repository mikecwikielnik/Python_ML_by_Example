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

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

# redo feature extraction using tfidf vectorizer & kmeans cluster algo
# on the resulting feature space

data = tfidf_vector.fit_transform(data_cleaned)

kmeans.fit(data)

clusters = kmeans.labels_

from collections import Counter
print(Counter(clusters))

# clustering here becomes more reasonable

# ex: take a closer look at the clusters by examining what they contain and
# w/ the top 10 terms (the terms with the highest tf-idf)

import numpy as np

cluster_label = {i: labels[np.where(clusters == i)] for i in range(k)}

terms = tfidf_vector.get_feature_names_out()
centroids = kmeans.cluster_centers_
for cluster, index_list in cluster_label.items():
    counter = Counter(cluster_label[cluster])
    print('cluster_{}: {} samples'.format(cluster, len(index_list)))
    for label_index, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print('{}: {} samples'.format(label_names[label_index], count))
    print('Top 10 terms:')
    for ind in centroids[cluster].argsort()[-10:]:
        print(' %s' % terms[ind], end="")
    print()
