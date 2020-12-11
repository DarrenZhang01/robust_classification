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
import tensorflow_datasets as tfds
from sklearn import datasets
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from gurobipy import *
import matplotlib.pyplot as plt
from utils import load_data

np.random.seed(1)


DATA_LIST = ["wine", "credit"]


for i, dataset in enumerate(DATA_LIST):
    plt.figure(i)
    X_train, X_test, Y_train, Y_test = load_data(dataset)
    x_axis = []
    y_axis = []

    #print(collections.Counter(Y_train))

    print(X_train.shape)
    print(Y_train)


    NUM_DATA = X_train.shape[0]
    NUM_FEATURES = X_train.shape[1]

    print("Now processing dataset: {}".format(dataset))
    print("# features: {}, # data points: {}".format(NUM_FEATURES, NUM_DATA))

    rho_list = [0.005]

    ################# Use Gurobi to train a Robust Logistic Regression #################
    for rho in rho_list:
        model = Model("logistic regression")
    
        beta = model.addVars(range(NUM_FEATURES), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0)
        beta_0 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
        l = model.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=0)
        t = model.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, obj=1)
    
        model.modelSense = GRB.MINIMIZE
    
        model.addConstrs(l[i] == -1 * Y_train[i] *
                         quicksum([beta[j] * X_train[i][j] + beta_0 for j in range(NUM_FEATURES)]) +
                         rho * quicksum([beta[k] * beta[k] for k in range(NUM_FEATURES)]) for i in range(NUM_DATA))
    
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
    
        acc = sum(Y_pred.flatten() == Y_test)/len(Y_test)
        
        x_axis.append(rho)
        y_axis.append(acc)
    plt.title("accuracy vs. robustness in Logistic Regression - {}".format(dataset))
    plt.plot(x_axis, y_axis)
    plt.savefig("LR_{}.png".format(dataset))




