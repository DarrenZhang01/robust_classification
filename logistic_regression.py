"""
An implementation of Robust logistic Regression.

2020.11.14

Reference: Bertsimas, D., Dunn, J., Pawlowski, C., & Zhuo, Y. D. (2019).
           Robust classification. INFORMS Journal on Optimization, 1(1), 2-34.
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn import datasets
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from gurobipy import *
import matplotlib.pyplot as plt
from utils import load_data

np.random.seed(1)


DATA_LIST = ["synthetic", "wine", "credit"]

for i, dataset in enumerate(DATA_LIST):
    plt.figure(i)
    X_train, X_test, Y_train, Y_test = load_data(dataset)
    x_axis = []
    y_axis = []

    NUM_DATA = X_train.shape[0]
    NUM_FEATURES = X_train.shape[1]

    print("Now processing dataset: {}".format(dataset))
    print("# features: {}, # data points: {}".format(NUM_FEATURES, NUM_DATA))

    if dataset == 'synthetic':
        rho_list = np.linspace(0, 0.5, 25)
    else:
        rho_list = np.linspace(0, 0.05, 25)

    ################# Use Gurobi to train a Robust Logistic Regression #################
    for rho in rho_list:
        model = Model("logistic regression")

        beta = model.addVars(range(NUM_FEATURES), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0)
        beta_0 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
        l = model.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
        t = model.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=1)
        beta_abs = model.addVars(range(NUM_FEATURES), vtype=GRB.CONTINUOUS, obj=0)

        model.modelSense = GRB.MINIMIZE
        model.addConstrs(beta_abs[i] == abs_(beta[i]) for i in range(NUM_FEATURES))
        model.addConstrs(l[i] == -1 * Y_train[i] *
                         quicksum([beta[j] * X_train[i][j] + beta_0 for j in range(NUM_FEATURES)]) +
                         rho * beta_abs.sum('*') for i in range(NUM_DATA))

        l_ = np.arange(-20., 20., 0.1).tolist()
        t_ = [math.log(1 + math.e ** i) for i in l_]

        for i in range(NUM_DATA):
            model.addGenConstrPWL(l[i], t[i], l_, t_)
        model.optimize()

        print("beta: {}".format(beta))
        print("beta_0: {}".format(beta_0))

        W_np = np.zeros(NUM_FEATURES)
        for j in range(NUM_FEATURES):
            W_np[j] = beta[j].x
        W_np = W_np.reshape((W_np.shape[0], 1))
        Y_pred = X_test @ W_np + beta_0.x
        Y_pred = np.where(Y_pred < 0, -1, Y_pred)
        Y_pred = np.where(Y_pred >= 0, 1, Y_pred)
        #Y_pred = 1/(1 + np.exp(-Y_pred))
        #Y_pred = Y_pred.reshape((Y_pred.shape[0],))
        acc = sum(Y_pred.flatten() == Y_test) / len(Y_test)
        #loss = hinge_loss(Y_test, Y_pred)
        x_axis.append(rho)
        y_axis.append(acc)
    plt.title("accuracy vs. robustness in Logistic Regression - {}".format(dataset))
    plt.plot(x_axis, y_axis)
    plt.xlabel('rho')
    plt.ylabel('accuracy')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig("LR_{}.png".format(dataset))
