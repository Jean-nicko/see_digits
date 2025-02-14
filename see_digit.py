import numpy as np
import pandas as pd

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes]
        self.weights = [np.random.randn(y, x)
                        for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def descente_gradient(self, training_data, iteration, mini_batch_size, eta, test_data=None):

        #manque implementation test data
        n = len(training_data)
        for j in range(iteration):
            np.random.shuffle(training_data)
            minibatch = [training_data[k:k+mini_batch_size]]




def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))