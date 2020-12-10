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
import matplotlib.pyplot as plt
from utils import load_data

np.random.seed(100)

# DATA_LIST = ["digit", "wine", "credit"]
DATA_LIST = ["wine", "credit"]


i = 0
for dataset in DATA_LIST:

  plt.figure(i)

  X_train, X_test, Y_train, Y_test = load_data(dataset)

  # Create two lists - "x_axis" and "y_axis" for storing the according
  # hyperparameter and the loss for each hyperparameter.
  x_axis = []
  y_axis = []

  print(collections.Counter(Y_train))

  print(X_train.shape)
  print(Y_train)


  NUM_DATA = X_train.shape[0]
  NUM_FEATURES = X_train.shape[1]

  print("Now processing dataset: {}".format(dataset))
  print("# features: {}, # data points: {}".format(NUM_FEATURES, NUM_DATA))

  rho_list = np.linspace(0, 0.005, 10)

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

    x_axis.append(rho)
    y_axis.append(loss)

  plt.title("loss vs. robustness in SVM - {}".format(dataset))
  plt.plot(x_axis, y_axis)
  plt.savefig("SVM_{}.png".format(dataset))

  i += 1
