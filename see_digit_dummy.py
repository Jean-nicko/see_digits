import pandas as pd
import numpy as np

data = pd.read_csv("C:/documents_local/`projet_math/data_csv/mnist_train.csv")
data_testing_complete = pd.read_csv("C:/documents_local/`projet_math/data_csv/mnist_test.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:2000].T
label_test = data_dev[0]
image_test = data_dev[1:785]

data_train = data[2000:60000].T
label_train = data_train[0]
image_train = data_train[1:785]

def init_w_b():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randon(10,1)
    return w1, b1, w2, b2

def forward_prog(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def reponse(y):
    reponse_y = np.zeros(y.size, y.max() + 1)
    reponse_y[np.arange(y.size, y)] = 1
    reponse_y = reponse_y.T
    return reponse_y


def backprog(z1, a1, z2, a2, w2, x, y):
    m = y.size
    reponse = reponse(y)
    dz2 = a2 - reponse
    dw2 = 1/ (m * np.dot(dz2, a1.T))
    db2 = 1/ (m * np.sum(dz2, 2))
    dz1 = np.dot(w2.T, dz2) * deriv_relu(z1)
    dw1 = 1 / (m * np.dot(dz1, x.T))
    db1 = 1 / (m * np.sum(dz1, 2))

    return dw1, db1, dw2, db2

def update_param(w1, b1, w2, b2, dw1, db1 ):


def relu(z):
    return np.maximum(0,z)

def deriv_relu(z):
    return z > 0


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))