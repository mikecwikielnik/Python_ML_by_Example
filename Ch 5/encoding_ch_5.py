"""
Converting categorical features to numerical – one-hot encoding and ordinal encoding

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 148). Packt Publishing. Kindle Edition. 
"""

from sklearn.feature_extraction import DictVectorizer

X_dict = [{'interest': 'tech', 'occupation': 'professional'},
          {'interest': 'fashion', 'occupatioin': 'student'},
          {'interest': 'fashion', 'occupation': 'professional'},
          {'interest': 'sports', 'occupation': 'student'},
          {'interest': 'tech', 'occupation': 'student'},
          {'interest': 'tech', 'occupation': 'retired'},
          {'interest': 'sports', 'occupation': 'professional'}]

dict_one_hot_encoder = DictVectorizer(sparse=False)
X_encoded = dict_one_hot_encoder.fit_transform(X_dict)
print(X_encoded)

# we can see the mapping by this line of code too

print(dict_one_hot_encoder.vocabulary_)

# when it comes to new data, we can tranform it with the following:

new_dict = [{'interest': 'sports', 'occupation': 'retired'}]
new_encoded = dict_one_hot_encoder.transform(new_dict)
print(new_encoded) # in the previous print, look at where they are ordered & u can match this code with that code

# we can inversely transform the encoded features back to the original features lik this:

print(dict_one_hot_encoder.inverse_transform(new_encoded))

# new category not encountered before

new_dict = [{'interest': 'unknown_interest', 'occupation': 'retired'},
            {'interest': 'tech', 'occupation': 'unseen_occupation'}]
new_encoded = dict_one_hot_encoder.transform(new_dict)
print(new_encoded)