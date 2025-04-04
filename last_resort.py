import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/documents_local/`projet_math/data_csv/mnist_train.csv")
datas = np.array(data)
label_train = datas[:, 0]
image_train = np.delete(datas, 0, 1)/255

w1 = np.random.random((15, 784)) - 0.5
b1 = np.random.random((15, 1)) - 0.5
w2 = np.random.random((10, 15)) - 0.5
b2 = np.random.random((10, 1)) - 0.5

minibatch = 100
epochs = 600
learning_rate = 1

def relu(x):
    return np.tanh(x/5)

def deriv_relu(x):
    return (1/5) * (1 - (np.tanh(x) ** 2))

def onehot(answer):
    m = np.zeros((1, 10))
    m[np.arange(1), answer] = 1
    return m

for i in range(epochs):

        correct = 0
        count = 1
        cost = 0
        w1average = np.zeros((15, 784))
        b1average = np.zeros((15, 1))
        w2average = np.zeros((10, 15))
        b2average = np.zeros((10, 1))

        #Forward Propagation
        for img, l in zip(image_train, label_train):
            answer = onehot(l).T
            ima = img.reshape(784, 1)

            z1 = np.dot(w1, ima) + b1
            a1 = relu(z1)
            z2 = np.dot(w2, a1) + b2
            a2 = z2

            if np.argmax(a2) == np.argmax(answer):
                correct += 1

            mse = 1/(2*count) * (np.subtract(a2, answer)) ** 2
            cost += mse
            deriv_error = (1/count) * np.subtract(a2, answer)

            nabla_weight2 = np.dot(deriv_error, a1.T)
            nabla_bias2 = deriv_error

            nabla_bias1 = np.multiply(np.dot(w2.T, deriv_error), deriv_relu(z1))
            nabla_weight1 = np.dot(nabla_bias1, ima.T)

            w1average += nabla_weight1
            b1average += nabla_bias1
            w2average += nabla_weight2
            b2average += nabla_bias2

            if count % minibatch == 0:
                w2 -= (learning_rate * w2average / count)
                b2 -= (learning_rate * b2average / count)
                w1 -= (learning_rate * w1average / count)
                b1 -= (learning_rate * b1average / count)




            count += 1

        print(f"Epoch #{i}: {correct / 600}  % accuracy, cost = {np.average(cost)}")






