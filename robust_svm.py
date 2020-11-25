"""
An implementation of Robust SVM based on MNIST dataset.

2020.11.14

Reference: Bertsimas, D., Dunn, J., Pawlowski, C., & Zhuo, Y. D. (2019).
           Robust classification. INFORMS Journal on Optimization, 1(1), 2-34.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from gurobipy import *

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

print("X_train: {}".format(X_train.shape))
print("Y_train: {}".format(Y_train.shape))
print("X_test: {}".format(X_test.shape))
print("Y_test: {}".format(Y_test))
# The shape of the data:
# X_train: (60000, 28, 28)
# Y_train: (60000,)
# X_test: (10000, 28, 28)
# Y_test: (10000,)

X_train = np.reshape(X_train, (60000, 784))
X_test = np.reshape(X_test, (10000, 784))
# Binarize label y such that it is equal to 1 when the digit is 1 and it is -1
# when the digit belongs to other classes.
Y_train_bin = np.where(Y_train < 1.5, 1, -1)
Y_test_bin = np.where(Y_test < 1.5, 1, -1)


NUM_DATA = 60000
NUM_FEATURES = 784

SVM = Model("robust_svm")

itas = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, obj=[1]*NUM_DATA)
W = SVM.addVars(range(NUM_FEATURES), vtype=GRB.CONTINUOUS, obj=[0]*NUM_FEATURES)
b = SVM.addVar(vtype=GRB.CONTINUOUS, obj=0)

SVM.modelSense = GRB.MINIMIZE

SVM.addConstrs([Y_train[i] * (quicksum([W[j] * X_train[i][j] for j in \
               range(NUM_FEATURES)]) - b) - quicksum([W[k] * W[k] for k in \
               range(NUM_FEATURES)]) >= 1 - itas[i] for i in range(NUM_DATA)])

SVM.optimize()
