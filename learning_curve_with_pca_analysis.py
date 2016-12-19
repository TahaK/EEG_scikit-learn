import pca
import plt_learning_curve
import numpy as np

# Author: Mustafa Taha Kocyigit -- <mustafataha93@gmail.com>

def analyze(estimator,X_ , y, range, name=""):

    X_ = pca.transform(X_, 300)

    # from sklearn import manifold
    # tsne = manifold.TSNE(n_components=1000, init='pca', random_state=0)
    # X_ = tsne.fit_transform(X_)
    for n in range:
        X = X_[:, 0:n+1]

        title = "Learning Curves" + name + " n=" + str(n)
        plt = plt_learning_curve.plot_learning_curve(estimator, title, X, y.ravel())

    plt.show()
