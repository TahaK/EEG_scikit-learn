import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.utils import shuffle

import explore_results
import learning_curve_with_pca_analysis
import pca

# Author: Mustafa Taha Kocyigit -- <mustafataha93@gmail.com>

X = pandas.read_csv('bu_data_432_446.csv', header=None).values
y = np.vstack((np.ones((432, 1)), np.zeros((446, 1))))
X = preprocessing.scale(X)

X, y = shuffle(X, y, random_state=0)

learning_curve_with_pca_analysis.analyze(linear_model.LogisticRegression(C=1e5), X, y,  [2, 10, 50, 100, 300])

# X = pca.transform(X, 50)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# logreg = linear_model.LogisticRegression(C=1e5)
# logreg.fit(X_train, y_train.ravel())
# y_pred = logreg.predict(X_test)
# print confusion_matrix(y_test, y_pred)
#
# target_names = ['ugly', 'beautiful']
# print(classification_report(y_test, y_pred, target_names=target_names))
#
# explore_results.plot_mash(logreg, X, y)
