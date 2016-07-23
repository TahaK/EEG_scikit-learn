import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn import tree

import explore_results
import learning_curve_with_pca_analysis
import pca
import plt_learning_curve

X = pandas.read_csv('bu_data_409_419.csv', header=None).values
y = np.vstack((np.ones((409, 1)), np.zeros((419, 1))))

X = np.nan_to_num(X)

X = preprocessing.scale(X)

print "Prepocessing"

# learning_curve_with_pca_analysis.analyze(tree.DecisionTreeClassifier(), X, y,  [2, 10, 50, 100, 300])

X = pca.transform(X, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train.ravel())

y_pred = decision_tree.predict(X_test)

target_names = ['beautiful', 'ugly']
print(classification_report(y_test, y_pred, target_names=target_names))

explore_results.plot_mash(decision_tree, X, y, h=0.05)

#
# dot_data = StringIO()
# tree.export_graphviz(decision_tree, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# # graph.write_pdf("tree.pdf")
