"""
Simulate a synthetic dataset such that we can pertube the data and test the
model robustness under feature perturbations.

2020.12.11
"""

import numpy as np

np.random.seed(100)
X = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=500)
print(X.shape)

Y = X[:,0] >= X[:,1]
Y = np.where(Y == True, 1, -1)
print(Y)

# Add potential pertubations.
# X = X + np.random.multivariate_normal(np.zeros(2), np.eye(2) * 0.1, size=500)


np.savetxt("X.txt", X)
np.savetxt("Y.txt", Y)
