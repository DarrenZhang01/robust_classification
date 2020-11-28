"""
An implementation of Robust logistic Regression based on MNIST dataset.
2020.11.14
Reference: Bertsimas, D., Dunn, J., Pawlowski, C., & Zhuo, Y. D. (2019).
           Robust classification. INFORMS Journal on Optimization, 1(1), 2-34.
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from gurobipy import *

np.random.seed(100)
#
# (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
#
# print("X_train: {}".format(X_train.shape))
# print("Y_train: {}".format(Y_train.shape))
# print("X_test: {}".format(X_test.shape))
# print("Y_test: {}".format(Y_test))
# # The shape of the data:
# # X_train: (60000, 28, 28)
# # Y_train: (60000,)
# # X_test: (10000, 28, 28)
# # Y_test: (10000,)
#
# X_train = np.reshape(X_train, (60000, 784))
# X_test = np.reshape(X_test, (10000, 784))
# # Binarize label y such that it is equal to 1 when the digit is 1 and it is -1
# # when the digit belongs to other classes.
# Y_train_bin = np.where(Y_train < 1.5, 1, -1)
# Y_test_bin = np.where(Y_test < 1.5, 1, -1)

iris = datasets.load_iris()
X = iris.data
Y = np.where(iris.target < 1.5, 1, -1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)



NUM_DATA = 105
NUM_FEATURES = 4
ro = 0.

##################### Use Gurobi to train a Robust SVM ########################

model = Model("logistic regression")

beta = model.addVars(range(NUM_FEATURES), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0)
beta_0 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
l = model.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
t = model.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=1)

model.modelSense = GRB.MINIMIZE

model.addConstrs(l[i] == -1 * Y_train[i] *
    quicksum([beta[j] * X_train[i][j] + beta_0 for j in range(NUM_FEATURES)]) + 
    ro * quicksum([beta[k] * beta[k] for k in range(NUM_FEATURES)]) for i in range(NUM_DATA))

l_ = np.arange(-20., 20., 0.1).tolist()
t_ = [math.log(1 + math.e ** i) for i in l_]
    
for i in range(NUM_DATA):
    model.addGenConstrPWL(l[i], t[i], l_, t_)
model.optimize()

print("beta: {}".format(beta))
print("beta_0: {}".format(beta_0))

beta_np = np.reshape(np.array([beta[0].x, beta[1].x, beta[2].x, beta[3].x]), (4, 1))
Y_pred = X_test @ beta_np + beta_0.x
Y_pred = np.where(Y_pred < 0, -1, Y_pred)
Y_pred = np.where(Y_pred >= 0, 1, Y_pred)

print(Y_pred.flatten() == Y_test)



