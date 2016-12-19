import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.decomposition import FastICA

from sklearn import svm, linear_model

X = pd.read_csv('stroop_data_698_698.csv', header=None)
y = np.vstack((np.ones((698, 1)), np.zeros((698, 1))))

rng = np.random.RandomState(4)

X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

X_train, X_cros, y_train, y_cros = train_test_split(X_, y_, test_size=0.25, random_state=rng)

ica = FastICA(random_state=rng, n_components=50, max_iter=10000)
X_ica = ica.fit(X_train).transform(X_train)

X_ica /= X_ica.std(axis=0)

# Preparing crossvalidation data for each component
X_cros_ica = ica.transform(X_cros)

X_cros_ica /= X_cros_ica.std(axis=0)

components = []

logreg = linear_model.LogisticRegression(C=1e5)

for i in range(0, 30):

    target_component = X_ica[:, i].reshape(X_ica.shape[0], 1)

    logreg.fit(target_component, y_train.ravel())

    X_cros_final = X_cros_ica[:, i].reshape(X_cros_ica.shape[0], 1)

    y_pred_cros = logreg.predict(X_cros_final)
    if accuracy_score(y_pred_cros, y_cros) > 0.53:
        components.append(i)

print "Remaining component counts : "+str(components.__len__())

# Reducing component counts
X_final = X_ica[:, components]

logreg.fit(X_final, y_train.ravel())

X_test_ica = ica.transform(X_test)

X_test_final = X_test_ica[:, components]

X_test_final /= X_test_final.std(axis=0)

y_pred = logreg.predict(X_test_final)

print accuracy_score(y_pred, y_test)