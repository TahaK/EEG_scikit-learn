from sknn.mlp import Classifier, Layer
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

import explore_results

nn = Classifier(
    layers=[
        Layer("Tanh", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

explore_results.plot_mash(nn, X, y)