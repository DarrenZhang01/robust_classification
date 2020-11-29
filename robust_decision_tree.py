"""
An implementation of Robust Decision Tree based on MNIST dataset.
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

iris = datasets.load_iris()
X = iris.data
Y = np.where(iris.target < 1.5, 1, -1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

NUM_DATA = 105
# Specify the number of nodes in the tree.
K = 7
# Specify the coefficients lambda for the variable d.
lambda_ = [1] * K

##################### Use Gurobi to train a Robust SVM ########################

model = Model("Robust Decision Tree")

# F is used to track the number of misclassified data points at k.
F = model.addVars(range(K), vtype=GRB.INTEGER, obj=1)
# Variable D is 1 if k is a leaf node otherwise 0.
D = model.addVars(range(K), vtype=GRB.BINARY, obj=[-l for l in lambda_])
# Variable Z to track which leaf node k at each data point in the training set is assigned.
Z = model.addVars(range(NUM_DATA), range(K), vtype=GRB.BINARY, obj=0)
# Use A and B to set splits for the tree.
A = model.addVars(range(K), range(4), lb=-GRB.INFINITY, vtype=GRB.BINARY, obj=0)
B = model.addVars(range(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0)
# Use G and H to count the number of points of the two labels in each node K.
G = model.addVars(range(K), vtype=GRB.INTEGER, obj=0)
H = model.addVars(range(K), vtype=GRB.INTEGER, obj=0)
# Add binary variables W, C
W = model.addVars(range(K), vtype=GRB.BINARY, obj=0)
C = model.addVars(range(K), vtype=GRB.BINARY, obj=0)


model.modelSense = GRB.MINIMIZE

model.addConstrs(G[k] == quicksum([(1 - Y_train[i]) * Z[i][k] / 2 for i in range(NUM_DATA)]) for k in range(K))
model.addConstrs(H[k] == quicksum([(1 + Y_train[i]) * Z[i][k] / 2 for i in range(NUM_DATA)]) for k in range(K))

model.addConstrs(F[k] <= G[k] + NUM_DATA * (W[k] + (1 - C[k])) for k in range(K))
model.addConstrs(F[k] <= H[k] + NUM_DATA * (1 - W[k] + 1 - C[k]) for k in range(K))
model.addConstrs(F[k] >= G[k] - NUM_DATA * (1 - W[k] + 1 - C[k]) for k in range(K))
model.addConstrs(F[k] >= H[k] - NUM_DATA * (W[k] + 1 - C[k]))
# The D value for the leave nodes must be equal to 1.
model.addConstrs(D[k] == 1 for k in range(math.floor(K / 2), K))
model.addConstrs(D[k] <= D[j] for k in [3, 4] for j in [0, 1])
model.addConstrs(D[k] <= D[j] for k in [5, 6] for j in [0, 2])
model.addConstrs(D[k] <= D[0] for k in [1, 2])
