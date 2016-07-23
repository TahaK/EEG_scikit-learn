from sklearn import svm
import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import explore_results
import learning_curve_with_pca_analysis
import pca
from sklearn.utils import shuffle

X = pandas.read_csv('bu_data_432_446.csv', header=None).values
y = np.vstack((np.ones((432, 1)), np.zeros((446, 1))))

X, y = shuffle(X, y, random_state=0)
#
# X = np.nan_to_num(X)
#
#
X = preprocessing.scale(X)
print "Prepocessing"

learning_curve_with_pca_analysis.analyze(svm.SVC(), X, y, [2, 10, 50, 100, 300])

# X = pca.transform(X, 2)
#
# X = preprocessing.scale(X)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#
# svm_classifier = svm.SVC()
# svm_classifier.fit(X_train, y_train.ravel())
# y_pred = svm_classifier.predict(X_test)
#
# print confusion_matrix(y_test, y_pred)
#
# target_names = ['ugly', 'beautiful']
# print(classification_report(y_test, y_pred, target_names=target_names))
#
# explore_results.plot_mash(svm_classifier, X, y, h=0.02)
