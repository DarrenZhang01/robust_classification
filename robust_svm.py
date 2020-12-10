"""
An implementation of Robust SVM based on MNIST dataset.

2020.11.14

Reference: Bertsimas, D., Dunn, J., Pawlowski, C., & Zhuo, Y. D. (2019).
           Robust classification. INFORMS Journal on Optimization, 1(1), 2-34.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn import datasets
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import collections
from gurobipy import *

np.random.seed(100)


def load_data(dataset):

  if dataset == "digit":

    digits = datasets.load_digits()
    X = digits.data
    Y = np.where(digits.target < 5, 1, -1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return (X_train, X_test, Y_train, Y_test)

  elif dataset == "wine":

    wine = tfds.load("wine_quality")
    train = wine["train"]
    X = np.zeros((4898, 11))
    Y = np.zeros(4898)
    i = 0
    for instance in tfds.as_numpy(train):
      X[i] = np.fromiter(instance["features"].values(), dtype=float)
      Y[i] = instance["quality"]
      i += 1
    X = normalize(X)
    Y = np.where(Y < 5, 1, -1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return (X_train, X_test, Y_train, Y_test)

  elif dataset == "credit":

    german_credit = tfds.load("german_credit_numeric")
    train = german_credit["train"]

    X = np.zeros((1000, 24))
    Y = np.zeros(1000)
    i = 0
    for instance in tfds.as_numpy(train):
      X[i] = instance["features"]
      Y[i] = instance["label"]
      i += 1
    X = normalize(X)
    Y = np.where(Y == 1, 1, -1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return (X_train, X_test, Y_train, Y_test)


X_train, X_test, Y_train, Y_test = load_data("credit")

print(collections.Counter(Y_train))

print(X_train.shape)
print(Y_train)


NUM_DATA = X_train.shape[0]
NUM_FEATURES = X_train.shape[1]

rho_list = np.linspace(0.0001, 0.005, 20)

########### Use Gurobi to train SVMs under different degrees of robustness #############
for rho in rho_list:

  SVM = Model("robust_svm")

  itas = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, obj=[0.0005]*NUM_DATA)
  W = SVM.addVars(range(NUM_FEATURES), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=[0]*NUM_FEATURES)
  b = SVM.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)

  SVM.modelSense = GRB.MINIMIZE
  SVM.Params.outputFlag = 0

  for i in range(NUM_DATA):
      SVM.addConstr(Y_train[i] * (quicksum([W[j] * X_train[i][j] for j in \
                    range(NUM_FEATURES)]) - b) - rho * quicksum([W[k] * W[k] for k in \
                    range(NUM_FEATURES)]) >= 1 - itas[i])

  SVM.optimize()

  W_np = np.zeros(NUM_FEATURES)
  for j in range(NUM_FEATURES):
    W_np[j] = W[j].x
  W_np = W_np.reshape((W_np.shape[0], 1))
  Y_pred = X_test @ W_np - b.x
  Y_pred = Y_pred.reshape((Y_pred.shape[0],))

  print("Y_test: {}, Y_pred: {}".format(Y_test.shape, Y_pred.shape))
  loss = hinge_loss(Y_test, Y_pred)
  print("the test loss for SVM under rho = {}: {}".format(rho, loss))
