"""
Best practice 9 – Deciding whether to reduce dimensionality, and if so, how to do so

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 358). Packt Publishing. Kindle Edition. 
"""

from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA 

# keep different number of top componenets
N = [10, 15, 25, 35, 45]
for n in N:
    pca = PCA(n_components=n)
    X_n_kept = pca.fit_transform(X)
    # estimate accuracy on the data set w/ top n components
    classifier = SVC(gamma=0.005)
    score_n_components = cross_val_score(classifier, X_n_kept, y).mean()
    print(f'Score with the dataset of top {n} components: {score_n_components:.2f}')

