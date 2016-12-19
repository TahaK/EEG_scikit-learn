from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import pandas
from sklearn import preprocessing

X = pandas.read_csv('bu_data_409_419.csv', header=None).values
y = np.vstack((np.ones((409, 1)), np.zeros((419, 1))))

# X = preprocessing.scale(X)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_ = tsne.fit_transform(X)
plt.scatter(X_[:, 0], X_[:, 1], c=y, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.show()

