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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import collections
from gurobipy import *
import matplotlib.pyplot as plt
from utils import load_data

np.random.seed(1)

DATA_LIST = ["synthetic", "wine", "credit"]


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
  
  if dataset == 'synthetic':
      rho_list = np.linspace(0, 0.5, 25)
  else:
      rho_list = np.linspace(0, 0.0002, 25)

  ########### Use Gurobi to train SVMs under different degrees of robustness #############
  for rho in rho_list:

    SVM = Model("robust_svm")

    itas = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, obj=[1]*NUM_DATA)
    W = SVM.addVars(range(NUM_FEATURES), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=[0]*NUM_FEATURES)
    b = SVM.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
    W_abs = SVM.addVars(range(NUM_FEATURES), vtype=GRB.CONTINUOUS, obj=0)

    SVM.modelSense = GRB.MINIMIZE
    SVM.Params.outputFlag = 0
    SVM.addConstrs(W_abs[i] == abs_(W[i]) for i in range(NUM_FEATURES))

    for i in range(NUM_DATA):
      SVM.addConstr(Y_train[i] * (quicksum([W[j] * X_train[i][j] for j in \
                    range(NUM_FEATURES)]) - b) - rho * W_abs.sum('*') >= 1 - itas[i])

    SVM.optimize()

    W_np = np.zeros(NUM_FEATURES)
    for j in range(NUM_FEATURES):
      W_np[j] = W[j].x
    W_np = W_np.reshape((W_np.shape[0], 1))
    Y_pred = X_test @ W_np - b.x
    # Y_pred = np.where(Y_pred > 0, 1, -1)
    Y_pred = Y_pred.reshape((Y_pred.shape[0],))
    Y_pred = np.where(Y_pred < 0, -1, Y_pred)
    Y_pred = np.where(Y_pred >= 0, 1, Y_pred)
    print("Y_test: {}, Y_pred: {}".format(Y_test.shape, Y_pred.shape))
    acc = sum(Y_pred.flatten() == Y_test) / len(Y_test)
    #loss = hinge_loss(Y_test, Y_pred)
    #print("the test hinge loss for SVM under rho = {}: {}".format(rho, loss))

    x_axis.append(rho)
    y_axis.append(acc)

  plt.title("accuracy vs. robustness in SVM - {}".format(dataset))
  plt.plot(x_axis, y_axis)
  plt.xlabel('rho')
  plt.ylabel('accuracy')
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
  plt.savefig("SVM_{}.png".format(dataset))

  i += 1
