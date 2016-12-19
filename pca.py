import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Author: Mustafa Taha Kocyigit -- <mustafataha93@gmail.com>

def transform(X, n):
    pca = PCA(n_components = n)

    pca.fit(X)

    print(pca.explained_variance_ratio_)
    print("Total variance explained by PCA : ")
    print np.sum(pca.explained_variance_ratio_)

    return pca.transform(X)


def plot():

    X__ = pandas.read_csv('stroop_data_250_263.csv', header=None).values
    y = np.vstack((np.ones((250, 1)), np.zeros((263, 1))))

    X__ = preprocessing.scale(X__)

    pca = PCA(n_components = 30)
    X_ = pca.fit_transform(X__)
    labels = X_[:,0] < 0;
    plt.scatter(X_[:, 0], X_[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.show()

    for i in range(0,513):
        if(labels[i] == True):
            if(i > 249):
                print i-250+1
            else:
                print i+1

# plot()