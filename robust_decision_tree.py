"""
An implementation of Robust Decision Tree based on MNIST dataset.
2020.11.14
Reference: Bertsimas, D., Dunn, J., Pawlowski, C., & Zhuo, Y. D. (2019).
           Robust classification. INFORMS Journal on Optimization, 1(1), 2-34.
"""

import numpy as np
from sklearn.decomposition import PCA
import math
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt
from gurobipy import *
from utils import load_data

np.random.seed(1)


# DATA_LIST = ["credit", "wine", "synthetic"]
DATA_LIST = ["wine", "credit"]


def find_all_parents(node):
    cur = node
    left, right = [], []
    while cur != 0:
        r = (cur-1)%2
        cur = (cur-1)//2
        if r == 0:
            left.append(cur)
        else:
            right.append(cur)
    return left, right


def get_acc(X_test, Y_test, node_labels):
    predictions = []
    for i in range(len(X_test)):
        test = X_test[i]
        cur_node = 0
        while D[cur_node].x == 0:
            W_np = np.zeros(NUM_FEATURES)
            for j in range(NUM_FEATURES):
                W_np[j] = A[cur_node, j].x
            if W_np.dot(test) < B[cur_node].x:
                cur_node = 2 * cur_node + 1
            else:
                cur_node = 2 * cur_node + 2
        predictions.append(node_labels[cur_node])
    return sum(Y_test == np.array(predictions))/len(Y_test)


def get_node_labels(X_train, Y_train):
    pred_count = {}
    for i in range(len(X_train)):
        cur_node = 0
        while D[cur_node].x == 0:
            W_np = np.zeros(NUM_FEATURES)
            for j in range(NUM_FEATURES):
                W_np[j] = A[cur_node, j].x
            if W_np.dot(X_train[i]) < B[cur_node].x:
                cur_node = 2 * cur_node + 1
            else:
                cur_node = 2 * cur_node + 2
        if cur_node not in pred_count:
            pred_count[cur_node] = [0, 0]
            if Y_train[i] == 1:
                pred_count[cur_node][1] += 1
            else:
                pred_count[cur_node][0] += 1
    for k in pred_count:
        count = pred_count[k]
        if count[0] > count[1]:
            pred_count[k] = -1
        else:
            pred_count[k] = 1
    return pred_count


for i, dataset in enumerate(DATA_LIST):
    plt.figure(i)
    X_train, X_test, Y_train, Y_test = load_data(dataset)
    if dataset in ["wine", "credit"]:
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    NUM_DATA = X_train.shape[0]
    NUM_FEATURES = X_train.shape[1]
    x_axis = []
    y_axis = []

    print("Now processing dataset: {}".format(dataset))
    print("# features: {}, # data points: {}".format(NUM_FEATURES, NUM_DATA))

    K = 7
    lambda_ = [0] * K
    N = 1
    epsilon = 0.01

    rho_list = np.linspace(0, 0.05, 10)
    for rho in rho_list:
        print(rho)
        ##################### Use Gurobi to train a Robust SVM ########################

        model = Model("Robust Decision Tree")

        # F is used to track the number of misclassified data points at k.
        F = model.addVars(range(K), vtype=GRB.INTEGER, obj=1)
        # Variable D is 1 if k is a leaf node otherwise 0.
        D = model.addVars(range(K), vtype=GRB.BINARY, obj=[l for l in lambda_])
        # Variable Z to track which leaf node k at each data point in the training set is assigned.
        Z = model.addVars(range(NUM_DATA), range(K), vtype=GRB.BINARY, obj=0)
        # Use A and B to set splits for the tree.
        A = model.addVars(range(K), range(NUM_FEATURES), lb=-GRB.INFINITY, vtype=GRB.BINARY, obj=0)
        B = model.addVars(range(K), lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0)
        # Use G and H to count the number of points of the two labels in each node K.
        G = model.addVars(range(K), vtype=GRB.INTEGER, obj=0)
        H = model.addVars(range(K), vtype=GRB.INTEGER, obj=0)
        # Add binary variables W, C
        W = model.addVars(range(K), vtype=GRB.BINARY, obj=0)
        C = model.addVars(range(K), vtype=GRB.BINARY, obj=0)


        model.modelSense = GRB.MINIMIZE
        model.Params.outputFlag = 0

        model.addConstrs(G[k] == quicksum([(1 - Y_train[i]) * Z[i, k] / 2 for i in range(NUM_DATA)]) for k in range(K))
        model.addConstrs(H[k] == quicksum([(1 + Y_train[i]) * Z[i, k] / 2 for i in range(NUM_DATA)]) for k in range(K))

        model.addConstrs(F[k] <= G[k] + NUM_DATA * (W[k] + (1 - C[k])) for k in range(K))
        model.addConstrs(F[k] <= H[k] + NUM_DATA * (1 - W[k] + 1 - C[k]) for k in range(K))
        model.addConstrs(F[k] >= G[k] - NUM_DATA * (1 - W[k] + 1 - C[k]) for k in range(K))
        model.addConstrs(F[k] >= H[k] - NUM_DATA * (W[k] + 1 - C[k]) for k in range(K))
        # The D value for the leave nodes must be equal to 1.
        model.addConstrs(D[k] == 1 for k in range(math.floor(K / 2), K))

        # The child leaves must have D values less than or equal to their parents.
        model.addConstrs(D[k] >= D[j] for k in [3, 4] for j in [0, 1])
        model.addConstrs(D[k] >= D[j] for k in [5, 6] for j in [0, 2])
        model.addConstrs(D[k] >= D[0] for k in [1, 2])


        # For the leaf nodes, the weights should be zero.
        model.addConstrs(D[k] + A.sum(k, "*") == 1 for k in range(K))
        # Each data point can only be assigned at a single leaf node.
        model.addConstrs(Z.sum(i, "*") == 1 for i in range(NUM_DATA))
        # Each data point cannot be assigned at the non-leaf nodes.
        model.addConstrs(Z[i, k] <= D[k] for i in range(NUM_DATA) for k in range(K))


        model.addConstrs(Z[i, k] <= 1 - D[j] for i in range(NUM_DATA) for k in [3, 4] for j in [0, 1])
        model.addConstrs(Z[i, k] <= 1 - D[j] for i in range(NUM_DATA) for k in [5, 6] for j in [0, 2])
        model.addConstrs(Z[i, k] <= 1 - D[0] for i in range(NUM_DATA) for k in range(1, 2))

        counts = Counter(Y_train)
        model.addConstrs(Z.sum('*', k) >= N * C[k] for k in range(K))
        # model.addConstrs(C[k] == D[k] for k in range(K))
        for k in range(K):
            model.addConstr(C[k] >= D[k] - quicksum([D[j] for j in (find_all_parents(k)[0] + find_all_parents(k)[1])]))
        model.addConstrs(quicksum([A[j, f] * X_train[i][f] for f in range(NUM_FEATURES)]) + rho +\
                        epsilon <= B[j] + NUM_DATA * (1 - Z[i, k]) for i in range(NUM_DATA) \
                        for k in range(K) for j in find_all_parents(k)[0])
        model.addConstrs(quicksum([A[j, f] * X_train[i][f] for f in range(NUM_FEATURES)]) - rho >= \
                        B[j] - NUM_DATA * (1 - Z[i, k]) for i in range(NUM_DATA) \
                        for k in range(K) for j in find_all_parents(k)[1])

        model.optimize()

        node_labels = get_node_labels(X_train, Y_train)
        acc = get_acc(X_test, Y_test, node_labels)
        x_axis.append(rho)
        y_axis.append(acc)

    plt.title("accuracy vs. robustness in Decision Tree - {}".format(dataset))
    plt.plot(x_axis, y_axis)
    plt.xlabel('rho')
    plt.ylabel('accuracy')
    plt.savefig("DT_{}.png".format(dataset))
