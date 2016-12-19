import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

import explore_results
import learning_curve_with_pca_analysis
import pca

# Author: Mustafa Taha Kocyigit -- <mustafataha93@gmail.com>

X = pandas.read_csv('bu_data_409_419.csv', header=None).values
y = np.vstack((np.ones((409, 1)), np.zeros((419, 1))))

print "Prepocessing"

# X = preprocessing.scale(X)

learning_curve_with_pca_analysis.analyze(LDA(), X, y, [2, 5, 10, 100, 1000])

# print(pca.explained_variance_ratio_)
# print("Total variance explained by PCA : ")
# print np.sum(pca.explained_variance_ratio_)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#
# lda = LDA()
# lda.fit(X_train, y_train.ravel())
# y_pred = lda.predict(X_test)
#
# target_names = ['ugly', 'beautiful']
# print(classification_report(y_test, y_pred, target_names=target_names))
#
# explore_results.plot_mash(lda, X, y)
