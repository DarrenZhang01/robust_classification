import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn import datasets
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


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