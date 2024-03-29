"""
Topic modeling using NMF

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 338). Packt Publishing. Kindle Edition. 
"""


from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


groups = fetch_20newsgroups(subset='all', categories=categories)

from nltk.corpus import names
all_names = set(names.words())



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

from sklearn.decomposition import NMF

t = 20
nmf = NMF(n_components=t, random_state=42)

# we specify 20 topics(n_components)

data = count_vector.fit_transform(data_cleaned)

# now, fit the model nmf on the term matrix data

nmf.fit(data)

# we can obtain the resulting topic-feature rank W after the model is trained

print(nmf.components_)

# for each topic, we display the top 10 terms based on their ranks

terms = count_vector.get_feature_names_out()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic {}:".format(topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

    