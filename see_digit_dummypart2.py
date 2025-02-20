import pandas as pd
import numpy as np

data = pd.read_csv("C:/mnist_train.csv")
data_testing_complete = pd.read_csv("C:/mnist_test.csv")

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
    b2 = np.random.randn(10,1)
    return w1, b1, w2, b2

def forward_prog(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    return z1, a1, z2, a2

def relu(z):
    return np.maximum(0,z)

def deriv_relu(z):
    return z > 0

def reponse(y):
    reponse_y = np.zeros(y.size, y.max() + 1)
    reponse_y[np.arange(y.size, y)] = 1
    reponse_y = reponse_y.T
    return reponse_y


def backprog(z1, a1, z2, a2, w2, x, y):
    m = y.size
    reponse = y.size
    C0 =2 * np.mean(a2 - reponse)
    dw2 = np.dot( C0 , np.dot(deriv_relu(z2).T, a1))
    db2 = C0 * deriv_relu(z2)
    dw1 = np.dot(C0 , np.dot(np.dot(deriv_relu(z1).T, np.dot(deriv_relu(z2),w2)).T, x))
    db1 = np.dot(C0.T , np.dot(deriv_relu(z1),np.dot(deriv_relu(z1).T, w2)))

    return dw1, db1, dw2, db2


def update_param(w1, b1, w2, b2, dw1, db1,dw2,db2, learning_rate):
    w1 -= dw1 * learning_rate
    w2 -= dw2 * learning_rate
    b1 -= db1 * learning_rate
    b2 -= db2 * learning_rate
    return w1, b1, w2, b2
