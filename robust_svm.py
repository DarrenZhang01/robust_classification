"""
An implementation of Robust SVM based on MNIST dataset.

2020.11.14

Reference: Bertsimas, D., Dunn, J., Pawlowski, C., & Zhuo, Y. D. (2019).
           Robust classification. INFORMS Journal on Optimization, 1(1), 2-34.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

print("X_train: {}".format(X_train.shape))
print("Y_train: {}".format(Y_train.shape))
print("X_test: {}".format(X_test.shape))
print("Y_test: {}".format(Y_test.shape))
