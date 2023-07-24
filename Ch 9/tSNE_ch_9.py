"""
Visualizing the newsgroups data with t-SNE

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 307). Packt Publishing. Kindle Edition. 
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']

groups_3 = fetch_20newsgroups(categories=categories_3)


from nltk.corpus import names
all_names = set(names.words())


count_vector_sw = CountVectorizer(stop_words="english", max_features=500)


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups_3.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)


data_cleaned_count_3 = count_vector_sw.fit_transform(data_cleaned)

from sklearn.manifold import _t_sne

\